import copy
import importlib
import inspect
import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from functools import wraps
from multiprocessing import Pool
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import joblib
import numpy as np
import pandas as pd
import pandas.api.types as pdt
import pyarrow as pa
from biocore import DataHandler, get_data_format
from biocore.utils.import_util import (
    is_biosets_available,
    is_datasets_available,
    is_polars_available,
    is_ray_available,
)
from biocore.utils.inspect import get_kwargs, get_required_args
from biocore.utils.naming import camelcase_to_snakecase
from biocore.utils.py_util import (
    is_bioset,
    is_dataset,
    is_dataset_dict,
    is_iterable_dataset,
)
from sklearn.utils.validation import (
    NotFittedError,
)

import biofit.config
from biofit.utils import (
    Unset,
    determine_upcast,
    fingerprint_from_data,
    fingerprint_from_kwargs,
    generate_cache_dir,
    get_cache_file_name,
    init_arrow_buffer_and_writer,
    logging,
    move_temp_file,
    update_fingerprint,
    version,
)
from biofit.utils.file_utils import expand_path, is_remote_url
from biofit.utils.fingerprint import Hasher, is_caching_enabled
from biofit.utils.py_util import iflatmap_unordered
from biofit.utils.table_util import string_to_arrow

if TYPE_CHECKING:
    from datasets.features.features import Features

logger = logging.get_logger(__name__)

T_CONFIG = TypeVar("T_CONFIG", bound="BaseConfig")
T_PROC = TypeVar("T_PROC", bound="BaseProcessor")

SelectedFeatureTypes = Union[Type, Tuple[Type], List[Union[Type, Tuple[Type]]]]

SelectedColumnTypes = Union[
    str,
    int,
    List[str],
    List[int],
    List[Union[List[str], List[int]]],
    SelectedFeatureTypes,
]


class NonExistentCacheError(Exception):
    """Used when we expect the existence of a cache"""

    pass


# based on conversion speed, we will use the following order of preference
_ORDERED_FORMATS = [
    "arrow",
    "pyarrow",
    "torch",
    "pt",
    "tf",
    "tensorflow",
    "pandas",
    "pd",
    "numpy",
    "np",
    "dicts",
    "dict",
    "list",
]

_DATAFRAME_FORMATS = [
    "pandas",
    "pd",
]

_SKLEARN_FORMATS = [
    "numpy",
    "np",
    "pandas",
    "pd",
    "series",
]

_SPECIAL_FORMATS = ["any", "all", "array", "dataframe", "sklearn"]

_ARROW_WRITEABLE_FORMATS = ["arrow", "pyarrow"]
# when https://github.com/huggingface/datasets/pull/6762 is merged, we ca add
# "polars", "pl" to the list of writeable formats

if is_polars_available():
    pandas_pos = _ORDERED_FORMATS.index("pandas")
    _ORDERED_FORMATS.insert(pandas_pos, "polars")
    _ORDERED_FORMATS.insert(pandas_pos + 1, "pl")

    _DATAFRAME_FORMATS.insert(0, "polars")
    _DATAFRAME_FORMATS.insert(1, "pl")

    _SKLEARN_FORMATS.insert(0, "polars")
    _SKLEARN_FORMATS.insert(1, "pl")


def sync_backup_config(func):
    @wraps(func)
    def wrapper(self, **kwargs):
        if hasattr(self, "config_"):
            self._fingerprint = None
            self.config_.replace_defaults(**kwargs)
        return func(self, **kwargs)

    return wrapper


def _generate_get_feature_names_out(estimator, n_features_out):
    """
    Modified _generate_get_feature_names_out from sklearn to convert estimator name to snakecase
    """

    def name_formatter(template, index):
        # Check if there is any placeholder {i...} in the template
        if re.search(r"\{i(\+([0-9]+))?\}", template) is None:
            # No placeholders, use default formatting {name}{i}
            template = f"{template}{index}"

        # Regex to find {i} or {i+n} where n is an integer
        pattern = r"\{i(\+([0-9]+))?\}"
        matches = re.finditer(pattern, template)

        for match in matches:
            full_match = match.group(0)
            increment = match.group(2)
            if increment:
                value = index + int(increment)
            else:
                value = index

            template = template.replace(full_match, str(value))

        return template

    estimator_name = None
    if hasattr(estimator, "config"):
        if getattr(estimator.config, "_feature_names_out", None) is not None:
            return np.asarray(estimator.config._feature_names_out)
        elif getattr(estimator.config, "output_template_name", None) is not None:
            estimator_name = estimator.config.output_template_name
        elif getattr(estimator.config, "processor_name", None):
            estimator_name = estimator.config.processor_name
            if n_features_out > 1:
                estimator_name += "_"
    if estimator_name is None:
        estimator_name = camelcase_to_snakecase(estimator.__class__.__name__)
        estimator_name = estimator_name.split("_for_")[0]
        estimator_name = "_".join(estimator_name.split("_")[:-1])
        if n_features_out > 1:
            estimator_name += "_"

    if n_features_out > 1:
        out = [name_formatter(estimator_name, i) for i in range(n_features_out)]
    else:
        out = [estimator_name]

    return np.asarray(out, dtype=object)


REPLAY_ACTIONS = [
    (
        "from_config_transform",
        {"path": "path", "fingerprint": "fingerprint", "ext": ".json"},
    ),
]


def post_recording(*args, **kwargs):
    out, _, func_args, func_kwargs = args
    if not hasattr(out, "_fingerprint") or not hasattr(out, "replays"):
        return out

    info = kwargs.get("info", None)

    if len(func_args) > 0:
        self: "BaseProcessor" = func_args[0]
        func_args = func_args[1:]
    else:
        self = func_kwargs["self"]
        del func_kwargs["self"]

    if len(func_args) > 0:
        ds = func_args[0]
        func_args = func_args[1:]
    else:
        ds = func_kwargs["X"]
        del func_kwargs["X"]

    out_fingerprint = None
    if info:
        out_fingerprint = info.get("new_fingerprint", None)

    if not out_fingerprint:
        out_fingerprint = getattr(out, "_fingerprint", None)

    fingerprint = self.fingerprint

    cache_file_names = []
    if info:
        cache_file_name = info.get("cache_file_name", None)
        if not cache_file_name:
            cache_dir = info.get("cache_dir", None)
            if cache_dir and os.path.exists(cache_dir) and fingerprint:
                file_list = os.listdir(cache_dir)
                cache_file_names = [
                    os.path.join(cache_dir, f)
                    for f in file_list
                    if fingerprint in f or (out_fingerprint and out_fingerprint in f)
                ]
        else:
            if cache_file_name and not isinstance(cache_file_name, list):
                cache_file_names = [cache_file_name]

    out._fingerprint = out_fingerprint
    replays = ds.replays or out.replays or []
    out.replays = replays.copy()
    entity_path = f"{self.__module__}.{self.__class__.__name__}"

    if cache_file_names:
        for func_name, replay_info in REPLAY_ACTIONS:
            path_arg = replay_info["path"]
            fingerprint_arg = replay_info["fingerprint"]
            ext = replay_info["ext"]
            paths = []
            for file in cache_file_names:
                if file.endswith(ext):
                    paths.append(file)

            if len(paths) == 1:
                new_kwargs = {path_arg: paths[0]}
                if fingerprint_arg:
                    new_kwargs[fingerprint_arg] = fingerprint
                out.replays.append(
                    (
                        out._fingerprint,
                        entity_path,
                        func_name,
                        (),
                        {**new_kwargs},
                    )
                )
                break

    return out


def keep_dataset_fingerprint(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) > 0:
            self = args[0]
            args = args[1:]
        else:
            self = kwargs["self"]
            del kwargs["self"]

        if len(args) > 0:
            ds = args[0]
            args = args[1:]
        else:
            ds = kwargs["X"]
            del kwargs["X"]

        old_fingerprint = getattr(ds, "_fingerprint", None)
        out = func(self, ds, *args, **kwargs)
        if old_fingerprint:
            ds._fingerprint = old_fingerprint
        return out

    return wrapper


@dataclass
class BaseConfig:
    @classmethod
    def from_config_file(
        cls, path: str, ignore_none=False, add_new_attr=True
    ) -> T_CONFIG:
        """
        Load configuration from a JSON file.

        Args:
            path (str or os.PathLike): Path to the configuration file.
            ignore_none (bool): If True, ignore None values during loading.
            add_new_attr (bool):
                If True, allow adding new attributes that are not explicitly defined.

        Returns:
            T_CONFIG:
                An instance of the configuration class with attributes loaded from the
                file.
        """
        if isinstance(path, os.PathLike):
            path = str(path)

        with open(path, "r") as f:
            states = json.load(f)
        return cls.from_dict(states, ignore_none=ignore_none, add_new_attr=add_new_attr)

    @classmethod
    def from_config(
        cls,
        config_or_path: Union[str, os.PathLike, dict, "BaseConfig"],
        ignore_none=False,
        add_new_attr=False,
    ) -> T_CONFIG:
        """
        Load configuration from a file, dictionary, or another BaseConfig instance.

        Args:
            config_or_path (Union[str, os.PathLike, dict, BaseConfig]):
                Configuration data or path.
            ignore_none (bool): If True, ignore None values during loading.
            add_new_attr (bool):
                If True, allow adding new attributes that are not explicitly defined.

        Returns:
            T_CONFIG: An instance of the configuration class.
        """
        if isinstance(config_or_path, (str, os.PathLike)):
            return cls.from_config_file(
                config_or_path, ignore_none=ignore_none, add_new_attr=add_new_attr
            )
        elif isinstance(config_or_path, dict):
            return cls.from_dict(
                config_or_path, ignore_none=ignore_none, add_new_attr=add_new_attr
            )
        elif isinstance(config_or_path, BaseConfig):
            return cls.from_dict(
                config_or_path.to_dict(deep=False),
                ignore_none=ignore_none,
                add_new_attr=add_new_attr,
            )
        else:
            raise ValueError(f"Unsupported config type {type(config_or_path)}")

    @classmethod
    def from_dict(cls, states, ignore_none=False, add_new_attr=False) -> T_CONFIG:
        """
        Load configuration from a dictionary.

        Args:
            states (dict): Dictionary containing configuration states.
            ignore_none (bool): If True, ignore None values during the assignment.
            add_new_attr (bool):
                If True, allow adding new attributes that are not in the init.

        Returns:
            T_CONFIG:
                An instance of the configuration class with attributes set according to
                `states`.
        """

        def _from_dict(obj):
            if isinstance(obj, dict):
                if len(obj) == 2 and "path" in obj and "format" in obj:
                    if obj["format"] == "joblib":
                        with open(obj["path"], "rb") as f:
                            return joblib.load(f)

                    return DataHandler.to_format(
                        obj["path"], target_format=obj["format"]
                    )
                elif "__module__" in obj and "__class__" in obj and "__dict__" in obj:
                    module = importlib.import_module(obj.get("__module__"))
                    _cls = getattr(module, obj["__class__"])
                    if not hasattr(_cls, "from_dict"):
                        raise ValueError(
                            f"from_dict method not found for class {_cls.__name__}"
                        )
                    return _cls.from_dict(obj["__dict__"])
                else:
                    return {k: _from_dict(v) for k, v in obj.items()}
            else:
                return obj

        _states = copy.deepcopy(states)
        cls_kwargs = get_kwargs(_states, cls.__init__)
        self = cls(
            **{
                k: _from_dict(v)
                for k, v in cls_kwargs.items()
                if not isinstance(v, Unset)
            }
        )

        for k in cls_kwargs:
            _states.pop(k, None)

        attributes = {k: _from_dict(v) for k, v in _states.items()}

        self = self.replace_defaults(
            ignore_none=ignore_none, add_new_attr=add_new_attr, **attributes
        )
        return self

    def to_dict(
        self,
        deep=True,
        save_nested_complex_obj=False,
        path=None,
        fingerprint=None,
    ):
        """
        Convert the configuration to a dictionary.

        Args:
            deep (bool): If True, recursively convert all attributes to dictionaries.
            save_nested_complex_obj (bool):
                If True, save complex nested objects to disk.
            path (str): Base path for saving complex nested objects.
            fingerprint (str):
                Optional fingerprint string to distinguish file paths.

        Returns:
            dict: A dictionary representation of the configuration.
        """
        base_dir = None
        if save_nested_complex_obj:
            if path:
                base_dir = os.path.dirname(path)
            else:
                base_dir = tempfile.mkdtemp()
                path = tempfile.NamedTemporaryFile("w", dir=base_dir, delete=False).name

        def convert_to_dict(obj, name=None):
            """
            Convert an object's attributes to a dictionary, recursively processing
            nested objects. Handles complex objects like arrays, datasets, and models
            by potentially saving them to disk and replaces them with a reference in
            the dictionary.

            Args:
                obj (Any): The object to be converted into a dictionary format.
                name (str, optional):
                    Optional name to be used for generating file paths when saving
                    complex objects. Helps in creating more readable and traceable file
                    names.

            Returns:
                dict:
                    A dictionary representing the object. For simple types, returns the
                    object itself. For complex objects, returns a dictionary with
                    metadata such as path and format or fully serialized object states.

            Raises:
                Exception:
                    If there is a problem with writing the data to disk or any other
                    operational issue during the conversion process, it raises an
                    Exception to indicate failure.

            Notes:
                This method deals with different types of objects including simple data
                types, complex machine learning models, and data structures. Depending
                on the 'save_nested_complex_obj' flag in the `to_dict` method, complex
                objects like machine learning models may either be saved to a file and
                represented by a path, or fully serialized into the dictionary.

                For saving data structures like arrays or datasets, it may use a
                temporary directory or a specified path to store intermediate files.
                This method handles the creation and cleanup of these files as needed.
            """
            if DataHandler.is_array_like(obj):
                fp = None
                _format = get_data_format(obj)
                arrow_writer_kwargs = {}
                if base_dir and save_nested_complex_obj:
                    if is_datasets_available():
                        from datasets import Features

                        if is_bioset(obj) or is_dataset(obj):
                            arrow_writer_kwargs["features"] = obj._info.features
                            obj = obj.data
                        else:
                            obj = DataHandler.to_format(obj, target_format="arrow")
                            arrow_writer_kwargs["features"] = (
                                Features.from_arrow_schema(obj.schema)
                            )
                    else:
                        obj = DataHandler.to_format(obj, target_format="arrow")
                        arrow_writer_kwargs["features"] = obj.schema
                    file_suffix = ""
                    if name:
                        file_suffix = f"-{name}"
                    new_fingerprint = update_fingerprint(
                        fingerprint or "", file_suffix + obj.__class__.__name__
                    )
                    fp = os.path.join(
                        base_dir, f"cache-{new_fingerprint}{file_suffix}.arrow"
                    )
                    arrow_writer_kwargs["fingerprint"] = new_fingerprint

                out = {
                    "path": fp,
                    "format": _format,
                }
                if base_dir and save_nested_complex_obj:
                    _, writer, tmp_file = init_arrow_buffer_and_writer(
                        cache_file_name=fp, **arrow_writer_kwargs
                    )
                    try:
                        if writer is not None:
                            writer.write_table(obj)
                            writer.finalize()
                        if tmp_file is not None:
                            move_temp_file(tmp_file, fp)

                    except (Exception, KeyboardInterrupt):
                        if writer is not None:
                            writer.finalize()
                        if tmp_file is not None:
                            tmp_file.close()
                            if os.path.exists(tmp_file.name):
                                os.remove(tmp_file.name)
                        raise
                return out
            if hasattr(obj, "to_dict") and callable(obj.to_dict):
                states = (
                    obj.to_dict(
                        save_nested_complex_obj=save_nested_complex_obj,
                        path=path,
                        fingerprint=fingerprint,
                    )
                    if isinstance(obj, BaseConfig)
                    else obj.to_dict()
                )
                return {
                    "__module__": obj.__class__.__module__,
                    "__class__": obj.__class__.__name__,
                    "__dict__": states,
                }
            if (
                "sklearn" in obj.__class__.__module__
                or "lightgbm" in obj.__class__.__module__
                or "xgboost" in obj.__class__.__module__
                or "catboost" in obj.__class__.__module__
            ):
                fp = None
                _format = "joblib"
                if base_dir and save_nested_complex_obj:
                    file_suffix = ""
                    if name:
                        file_suffix = f"-{name}"
                    new_fingerprint = update_fingerprint(
                        fingerprint or "", file_suffix + obj.__class__.__name__
                    )
                    fp = os.path.join(
                        base_dir, f"cache-{new_fingerprint}{file_suffix}.joblib"
                    )

                    with open(fp, "wb") as f:
                        joblib.dump(obj, f)

                return {
                    "path": fp,
                    "format": _format,
                }

            if pdt.is_dict_like(obj):
                return {k: convert_to_dict(v, k) for k, v in obj.items()}
            if pdt.is_list_like(obj):
                return [convert_to_dict(v) for v in obj]
            if isinstance(obj, (str, bool, type(None))):
                return obj
            if pdt.is_integer(obj):
                return int(obj)
            if pdt.is_number(obj):
                return float(obj)

        if deep:
            if hasattr(self, "__getstate__"):  # python>=3.11
                states = copy.deepcopy(
                    convert_to_dict(
                        {
                            k: v
                            for k, v in self.__getstate__().items()
                            if not isinstance(v, type)
                        }
                    )
                )
            else:
                states = copy.deepcopy(
                    convert_to_dict(
                        {
                            k: v
                            for k, v in self.__dict__.items()
                            if not isinstance(v, type)
                        }
                    )
                )
        else:
            states = copy.deepcopy(self.__dict__)

        states["config_name"] = self.__class__.__name__
        return states

    def save_to_cache(self, path, fingerprint=None):
        """
        Save the configuration to a cache file in JSON format.

        Args:
            path (str): Path where the configuration will be saved.
            fingerprint (str): Optional fingerprint string to distinguish file paths.
        """
        base_dir = os.path.dirname(path)
        os.makedirs(base_dir, exist_ok=True)
        files = os.listdir(base_dir)
        try:
            states = self.to_dict(
                path=path, save_nested_complex_obj=True, fingerprint=fingerprint
            )
        except (Exception, KeyboardInterrupt):
            new_files = set(os.listdir(base_dir)) - set(files)
            for f in new_files:
                if os.path.exists(f):
                    os.remove(f)
            raise

        with open(path, "w") as f:
            json.dump(states, f, check_circular=False)

    def replace_defaults(
        self,
        ignore_none=False,
        add_new_attr=False,
        return_unused_kwargs=False,
        **states,
    ):
        """
        Replace default values of the config instance with provided values.

        Args:
            ignore_none (bool): If True, ignore None values during the replacement.
            add_new_attr (bool):
                If True, allow adding new attributes that are not explicitly defined.
            return_unused_kwargs (bool):
                If True, return unused keyword arguments.

        Returns:
            self or (self, dict):
                The configuration instance or a tuple of the instance and unused
                kwargs.
        """
        unused_keys = []
        for k, v in states.items():
            if isinstance(v, Unset):
                continue
            if add_new_attr or hasattr(self, k):
                if not ignore_none or v is not None:
                    setattr(self, k, v)
            else:
                unused_keys.append(k)
        if return_unused_kwargs:
            return self, {k: states[k] for k in unused_keys}
        return self

    def _repr_mimebundle_(self, **kwargs):
        """
        Return the MIME bundle for the representation of the estimator.

        This function is utilized by Jupyter environments to display the estimator.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            dict: A MIME type bundle representing the estimator.
        """
        from sklearn._config import get_config
        from sklearn.utils._estimator_html_repr import estimator_html_repr

        output = {"text/plain": repr(self)}
        if get_config()["display"] == "diagram":
            output["text/html"] = estimator_html_repr(self)
        return output

    def get_params(self, deep=True, show_init_only=True, show_repr_only=True):
        """
        Get parameters for the estimator.

        Args:
            deep (bool): If True, return parameters of nested objects.

        Returns:
            dict: Dictionary of parameters.
        """
        params = {}
        args = [
            f.name
            for f in fields(self)
            if (not show_init_only or f.init) and (not show_repr_only or f.repr)
        ]
        for param in args:
            obj = getattr(self, param, "not_found")
            if isinstance(obj, str) and obj == "not_found":
                # check in dataclass fields for default values
                for f in fields(self):
                    if f.name == param:
                        if f.default_factory is not None:
                            obj = f.default_factory()
                        else:
                            obj = f.default
                        break
            if hasattr(obj, "get_params") and deep:
                params[param] = obj.get_params(deep=deep)
            elif not pdt.is_complex(obj) or deep:
                params[param] = obj
        return params


def get_processor_from_config_name(
    config_name, processor_type=None, processor_name=None
):
    try:
        package = "biofit"
        module_name = "models"
        if processor_type:
            package = f"{package}.{module_name}"
            module_name = processor_type
            if processor_name:
                package = f"{package}.{module_name}"
                module_name = processor_name
        package = package.replace("-", "_")
        module_name = module_name.replace("-", "_")
        module = importlib.import_module(f".{module_name}", package=package)
        config_cls = getattr(module, config_name)
    except (ModuleNotFoundError, AttributeError):
        return None

    return config_cls


@dataclass
class FitTransformConfig(BaseConfig):
    # common attributes

    map_kwargs: dict = field(default_factory=lambda: {"fn_kwargs": {}})
    version: str = "0.0.0"

    # only here to transmit the values to the next config
    load_from_cache_file = is_caching_enabled()

    def populate_map_kwargs(self):
        if "keep_in_memory" not in self.map_kwargs:
            self.map_kwargs["keep_in_memory"] = not self.cache_output

        if "cache_file_name" not in self.map_kwargs:
            self.map_kwargs["cache_file_name"] = self.cache_file_name

        if "num_proc" not in self.map_kwargs:
            self.map_kwargs["num_proc"] = self.num_proc

        return self

    @classmethod
    def prepare_config(
        cls,
        **kwargs,
    ):
        self = cls()
        self = self.replace_defaults(**kwargs)
        self = self.populate_map_kwargs()
        return self


@dataclass
class FitConfig(FitTransformConfig):
    # to be used for ray data only
    concurrency: Union[Tuple[int, int], int] = None


@dataclass
class TransformConfig(FitConfig):
    @classmethod
    def from_config(cls, config: FitTransformConfig):
        if isinstance(config, FitTransformConfig):
            states = copy.deepcopy(config.__dict__)
        elif isinstance(config, dict):
            states = copy.deepcopy(config)
        else:
            return super().from_config(copy.deepcopy(config))
        return cls.prepare_config(**states)


@dataclass
class ProcessorConfig(BaseConfig):
    f"""
    Stores the parameters for all processors.

    The config should contain all the parameters for transforming the data without the
    need to repeat fitting.

    Args:
        output_format (str, *optional*):
            The output format of the transformed data. The format will be the same as
            the input data if not provided. Possible values are {_ORDERED_FORMATS}.
        input_columns (_SelectedColumnTypes, *optional*):
            The input columns to be used for fitting the processor and transforming the
            data. If more than one table or array is provided, a list of lists will
            correspond to each input argument. A single list will be applied to only
            the first input argument. Only `datasets.Bioset`,
            `datasets.IterableDataset`, or `biofit.Bioset` input data support column
            name selection via `datasets.Features` object.
        unused_columns (_SelectedColumnTypes, *optional*):
            The columns that are not used for fitting the processor. This is ignored if
            `input_columns` is provided. A single list or item will be applied to all
            input arguments, while a list of lists will correspond to each input
            argument.
        keep_unused_columns (bool):
            Whether to keep the unused columns in the final output. Default is `True`.
        raise_if_missing (bool):
            Whether to raise an error if the input columns provided are missing during
            fitting. Default is `True`. If `False`, the processor will use the columns
            that are present in the input data and ignore the missing columns. Use this
            when the pipeline contains feature selection before the processor.
        enable_caching (bool, *optional*):
            Whether to disable or enable writing and reading from cache. Default is
            `True`.
        cache_dir (str, *optional*):
            The directory to store the cached data. Default is `None`.
        version (str):
            The version of the processor. Default is `"0.0.0"`. This is used for
            caching purposes. Set this to a new version when the processor is updated
            and the parameters are the same as a previous version.

    Attributes:
        _fit_process_desc (str):
            The description next to the progress bar during fitting.
        _transform_process_desc (str):
            The description next to the progress bar during transformation.
        _input_feature_types (_SelectedFeatureTypes, *optional*):
            The input feature types that will be applied by the processor. Only used
            when input has a `features` attribute containing `datasets.Features`, such
            as a `datasets.Bioset`, `datasets.IterableDataset`, or `biofit.Bioset`.
            When `input_columns` is provided, this attribute is ignored.
        _unused_feature_types (_SelectedFeatureTypes, *optional*):
            The feature types that are not used for fitting the processor. This is
            ignored if `input_columns` is provided.
        features_out_suffix (str, *optional*):
            The suffix to be added to the output feature names.
        features_out_prefix (str): The prefix to be added to the output feature names.
        processor_type (str, *optional*):
            The type of the processor (e.g. feature_selection, scaling, imputation,
            etc.). Used for auto class instantiation. Must be the same as the name of
            the parent module where the processor is defined. A `None` value implies
            that the processor has no parent module.
        processor_name (str):
            The name of the processor (e.g. select_k_best, min_max_scaler,
            simple_imputer, etc.). Used for auto class instantiation. Must be the same
            as the module name where the processor is defined.
        experiment_name (str, *optional*):
            The name of the dataset the processor is applied to. Used for auto class
            instantiation based on the type of the dataset. A `None` value implies that
            the processor is not dataset-specific.
        output_template_name (str, *optional*):
            The name of the output template. This is used to generate the output
            feature names.
        is_fitted (bool): Whether the processor is fitted to the input data.
        _batch_method_prefix (str):
            The prefix to be added to the batch method name. This is used to call the
            batch method during fitting.
        _input_columns (List[_SelectedColumnTypes], *optional*):
            The parsed input columns, with number of lists equaling to the number of
            input arguments. This is generated from
            `TransformationMixin._set_input_columns_and_arity`.
        n_features_in_ (int): The number of input features.
        _n_features_out (int, *optional*):
            The number of output features. Anything other than `None` implies that the
            processor is not one-to-one. Set this during fit or before transformation
            only if the transformation results in new features *and* the number of
            output features is not equal to the number of input features. For example,
            feature extraction, feature generation, etc. Preprocessing steps like
            feature selection does not result in *new* features and should be kept as
            `None`.
        _features_out (Features, *optional*):
            The `datasets.Features` object for the output features. This is inferred
            before transformation. Used for caching the output data as an arrow ipc
            stream.
        feature_idx_in_ (List[int], *optional*):
            The column indices of the input that were used for fitting. This is
            automatically set.
        feature_names_in_ (List[str], *optional*):
            The column names of the input features. This is automatically set during
            fitting. Only set if the input data during fit supports column names.
            Transformations will use this attribute to select the input columns, if
            supported. Otherwise, `feature_idx_in_` will be used.
        target_idx_in_ (List[int], *optional*):
            The column indices of the target that were used for fitting. This is
            automatically set.
        target_name_in_ (List[str], *optional*):
            The column names of the target features. This is automatically set during
            fitting. Only set if the target data during fit supports column names.
            Transformations will use this attribute to select the target columns, if
            supported. Otherwise, `target_idx_in_` will be used.
        one_to_one_features (bool):
            Whether the transformation results in a one-to-one mapping of input to
            output features. This will be `True` if `_n_features_out` is `None` and
            `False` otherwise.
        _returns_tuple (bool):
            Whether the transform method returns a tuple of data. See
            `MinPrevalenceRowSampleFilter` for an example.
        _data_fingerprint (str):
            The fingerprint of the input data. This is used to recognize the input data
            during transformation. If the input is the same as the data used for
            fitting, information from the fit process is reused to process the input
            data for transformation (e.g. selecting the same columns, etc.).
    """

    output_format: str = field(default=None, init=True, repr=False)
    input_columns: SelectedColumnTypes = field(default=None, init=True, repr=False)
    unused_columns: SelectedColumnTypes = field(default=None, init=True, repr=False)
    keep_unused_columns: bool = field(default=True, init=True, repr=False)
    raise_if_missing: bool = field(default=True, init=True, repr=False)
    enable_caching: bool = field(default=True, init=True, repr=False)
    cache_output: bool = field(default=True, init=True, repr=False)
    load_from_cache_file: bool = field(default=True, init=True, repr=False)
    cache_dir: str = field(default=None, init=True, repr=False)
    version: str = field(default=version.__version__, init=True, repr=True)

    _fit_process_desc: str = field(
        default="Fitting the processor to the input data", init=False, repr=False
    )
    _transform_process_desc: str = field(
        default="Transforming the input data", init=False, repr=False
    )

    _fit_input_feature_types: SelectedFeatureTypes = field(
        default=None, init=False, repr=False
    )
    _transform_input_feature_types: SelectedFeatureTypes = field(
        default=None, init=False, repr=False
    )
    _fit_unused_feature_types: SelectedFeatureTypes = field(
        default=None, init=False, repr=False
    )
    _transform_unused_feature_types: SelectedFeatureTypes = field(
        default=None, init=False, repr=False
    )

    _input_columns: SelectedColumnTypes = field(default=None, init=False, repr=False)
    features_out_suffix: str = field(default=None, init=False, repr=False)
    features_out_prefix: str = field(default=None, init=False, repr=False)
    processor_type: str = field(default="", init=False, repr=False)
    processor_name: str = field(default="", init=False, repr=False)
    experiment_name: str = field(default="", init=False, repr=False)

    output_template_name: str = field(default=None, init=False, repr=False)

    # automatically generated attributes
    _batch_method_prefix: str = field(default="_partial", init=False, repr=False)
    is_fitted: bool = field(default=False, init=False, repr=False)
    n_features_in_: int = field(default=None, init=False, repr=False)
    _n_features_out: int = field(default=None, init=False, repr=False)
    _features_out: list = field(default=None, init=False, repr=False)
    feature_idx_in_: List[int] = field(default=None, init=False, repr=False)
    feature_names_in_: List[str] = field(default=None, init=False, repr=False)
    extra_idx_in_: List[List[int]] = field(default=None, init=False, repr=False)
    extra_names_in_: List[List[str]] = field(default=None, init=False, repr=False)
    _feature_names_out: List[str] = field(default=None, init=False, repr=False)
    _feature_idx_out: List[int] = field(default=None, init=False, repr=False)
    _returns_tuple: bool = field(default=False, init=False, repr=False)
    _data_fingerprint: str = field(default=None, init=False, repr=False)

    @property
    def one_to_one_features(self):
        return self._n_features_out is None

    def to_dict(self, *args, deep=True, **kwargs):
        states = super().to_dict(*args, deep=deep, **kwargs)
        states["_fit_process_desc"] = self._fit_process_desc
        states["_transform_process_desc"] = self._transform_process_desc
        if deep:

            def set_nested_feature_type(ft_name):
                ft_obj = getattr(self, ft_name)
                if isinstance(ft_obj, type):
                    states[ft_name] = [ft_obj.__name__]
                elif isinstance(ft_obj, tuple):
                    states[ft_name] = [tuple(ft.__name__ for ft in ft_obj)]
                elif isinstance(ft_obj, list):
                    states[ft_name] = []
                    for ft in ft_obj:
                        if isinstance(ft, type):
                            states[ft_name].append(ft.__name__)
                        elif isinstance(ft, tuple):
                            states[ft_name].append(tuple(f.__name__ for f in ft))
                        elif ft is None:
                            states[ft_name].append(None)
                elif ft_obj is None:
                    states[ft_name] = None

            set_nested_feature_type("_fit_input_feature_types")
            set_nested_feature_type("_transform_input_feature_types")
            set_nested_feature_type("_fit_unused_feature_types")
            set_nested_feature_type("_transform_unused_feature_types")

        states["features_out_suffix"] = self.features_out_suffix
        states["features_out_prefix"] = self.features_out_prefix
        states["processor_name"] = self.processor_name
        states["processor_type"] = self.processor_type
        states["experiment_name"] = self.experiment_name
        return states

    @classmethod
    def from_dict(
        cls, states: dict, ignore_none: bool = False, add_new_attr: bool = False
    ) -> T_CONFIG:
        self = super().from_dict(
            states, ignore_none=ignore_none, add_new_attr=add_new_attr
        )

        if is_datasets_available():
            from datasets.features.features import _FEATURE_TYPES
        else:
            _FEATURE_TYPES = {}

        def get_nested_feature_type(ft):
            if isinstance(ft, str):
                if not is_datasets_available():
                    raise ValueError(
                        "Trying to load cache using datasets.Feature without datasets "
                        "installed. Please install datasets to load the cache."
                    )
                return _FEATURE_TYPES.get(ft)
            elif isinstance(ft, (tuple, list)):
                return tuple(get_nested_feature_type(f) for f in ft)
            return ft

        if self._fit_input_feature_types:
            if not isinstance(self._fit_input_feature_types, list):
                self._fit_input_feature_types = [self._fit_input_feature_types]
            self._fit_input_feature_types = [
                get_nested_feature_type(ft) for ft in self._fit_input_feature_types
            ]
        if self._fit_unused_feature_types:
            if not isinstance(self._fit_unused_feature_types, list):
                self._fit_unused_feature_types = [self._fit_unused_feature_types]
            self._fit_unused_feature_types = [
                get_nested_feature_type(ft) for ft in self._fit_unused_feature_types
            ]
        if self._transform_input_feature_types:
            if not isinstance(self._transform_input_feature_types, list):
                self._transform_input_feature_types = [
                    self._transform_input_feature_types
                ]
            self._transform_input_feature_types = [
                get_nested_feature_type(ft)
                for ft in self._transform_input_feature_types
            ]
        if self._transform_unused_feature_types:
            if not isinstance(self._transform_unused_feature_types, list):
                self._transform_unused_feature_types = [
                    self._transform_unused_feature_types
                ]
            self._transform_unused_feature_types = [
                get_nested_feature_type(ft)
                for ft in self._transform_unused_feature_types
            ]
        return self

    def replace_defaults(self, ignore_none=False, add_new_attr=False, **states):
        for k, v in states.items():
            if isinstance(v, Unset):
                continue
            if (add_new_attr or hasattr(self, k)) and (
                not ignore_none or v is not None
            ):
                setattr(self, k, v)
        return self


class TransformationMixin:
    def _get_method(self, formats, func_type, prefix=None):
        """
        Retrieves processing methods based on the function type and target format.

        Args:
            format (str): The target format.
            func_type (str): The type of processing method.
            prefix (str, *optional*):
                The prefix to be added to the method name (e.g "_partial" for batch
                processing methods). Default is None.

        Returns:
            A list of processing methods based on the target format.
        """
        funcs = []

        if isinstance(formats, str):
            format = [formats]
        for format in formats + _SPECIAL_FORMATS:
            if prefix:
                func = getattr(self, f"{prefix}{func_type}_{format}", None)
                if func is not None:
                    funcs.append(func)
            else:
                func = getattr(self, f"{func_type}_{format}", None)
                if func is not None:
                    funcs.append(func)
        return funcs

    def _has_method(self, formats, func_type, prefix=None):
        """
        Checks if atleast one method exists for the given [prefix]_[func_type]_[format]
        or [func_type]_[format] if prefix is not given.

        Args:
            format (str): The target format.
            func_type (str): The type of processing method.
            prefix (str, *optional*):
                The prefix to be added to the method name (e.g "_partial" for batch
                processing methods). Default is None.
            check_only (bool, *optional*):
                If True, only checks if a method exists. Default is False.

        Returns:
            True if a method exists for any of the given formats, False
            otherwise.
        """

        if isinstance(formats, str):
            format = [formats]
        for format in formats + _SPECIAL_FORMATS:
            if prefix:
                func = getattr(self, f"{prefix}{func_type}_{format}", None)
                if func is not None:
                    return True
            else:
                func = getattr(self, f"{func_type}_{format}", None)
                if func is not None:
                    return True
        return False

    def _get_target_func(
        self,
        funcs,
        source_format,
        target_formats=None,
        accepted_formats=_ORDERED_FORMATS,
    ):
        formats = []
        new_funcs = []
        for f in accepted_formats:
            for fun in funcs:
                if fun.__name__.endswith(f"_{f}"):
                    formats.append(f)
                    new_funcs.append(fun)

        if formats:
            # this class has a method for the target format
            to_format = formats[0]
            if source_format in formats:
                to_format = source_format
            if target_formats is not None:
                if not isinstance(target_formats, list):
                    target_formats = [target_formats]
                for target_format in target_formats:
                    if target_format in formats:
                        to_format = target_format
                        break
                else:
                    logger.warning(
                        f"Using {self.__class__.__name__} using `{target_formats}` is "
                        f"not supported. Formatting input to `{to_format}` instead"
                    )

            return new_funcs[formats.index(to_format)], to_format

        any_funcs = [
            f
            for f in funcs
            if f.__name__.endswith("_any") or f.__name__.endswith("_all")
        ]
        if any_funcs:
            return any_funcs[0], source_format

        tbl_funcs = [f for f in funcs if f.__name__.endswith("_array")]
        if tbl_funcs:
            target_formats = _ORDERED_FORMATS
            funcs = tbl_funcs
        else:
            df_funcs = [f for f in funcs if f.__name__.endswith("_dataframe")]
            if df_funcs:
                target_formats = _DATAFRAME_FORMATS
                funcs = df_funcs
            else:
                arr_funcs = [f for f in funcs if f.__name__.endswith("_sklearn")]
                if arr_funcs:
                    target_formats = _SKLEARN_FORMATS
                    funcs = arr_funcs

        if funcs and len(funcs):
            func = funcs[0]
            to_format = None
            if target_formats is not None and len(funcs):
                # we assume that the first function handles all formats within fn_trans_format
                # e.g sklearn functions with [polars, numpy, pandas] formats
                if isinstance(target_formats, list):
                    # prioritize input format over output format if both are supported
                    if source_format in target_formats:
                        to_format = source_format
                    else:
                        to_format = target_formats[0]

            return func, to_format

        return None, target_formats

    def _parse_column_selection(self, *args, from_config=False):
        if from_config:
            return self._parse_column_selection_from_config(*args)
        return self._parse_column_selection_from_self(*args)

    def _parse_column_selection_from_config(self, *args, from_config=False):
        out = {}
        if len(args):
            if self.config.feature_names_in_:
                out[args[0]] = self.config.feature_names_in_
            else:
                out[args[0]] = [f"{i}" for i in self.config.feature_idx_in_]
        if len(args) > 1:
            for i, arg in enumerate(args):
                if (
                    self.config.extra_names_in_
                    and len(self.config.extra_names_in_) > i
                    and self.config.extra_names_in_[i]
                ):
                    out[arg] = self.config.extra_names_in_[i]
                elif (
                    self.config.extra_idx_in_
                    and len(self.config.extra_idx_in_) > i
                    and self.config.extra_idx_in_[i]
                ):
                    out[arg] = [f"{j}" for j in self.extra_idx_in_[i]]
        return out

    def _parse_column_selection_from_self(self, *args, from_config=False):
        out = {}
        if len(args):
            if self.feature_names_in_:
                out[args[0]] = self.feature_names_in_
            else:
                out[args[0]] = [f"{i}" for i in self.feature_idx_in_]
        if len(args) > 1:
            for i, arg in enumerate(args):
                if (
                    self.extra_names_in_
                    and len(self.extra_names_in_) > i
                    and self.extra_names_in_[i]
                ):
                    out[arg] = self.extra_names_in_[i]
                elif (
                    self.extra_idx_in_
                    and len(self.extra_idx_in_) > i
                    and self.extra_idx_in_[i]
                ):
                    out[arg] = [f"{j}" for j in self.extra_idx_in_[i]]
        return out

    def _set_input_columns_and_arity(self, *args):
        input_columns = None
        if len(args) > 1:
            input_columns = [None] * len(args)
            for i, arg in enumerate(args):
                if arg is not None:
                    if isinstance(arg, (str, int)):
                        input_columns[i] = [arg]
                    elif isinstance(arg, list):
                        input_columns[i] = arg
        else:
            input_columns = args[0] or None
            if isinstance(input_columns, (str, int)):
                input_columns = [input_columns]
            input_columns = [input_columns]
        return input_columns

    def _reinsert_columns(
        self, input, out, indices, unused_indices, one_to_one_features=False
    ):
        out_dims = DataHandler.get_shape(out)
        x_dims = DataHandler.get_shape(input)

        if unused_indices and x_dims[0] == out_dims[0]:
            other_col_names = DataHandler.get_column_names(input, generate_cols=True)
            other_col_names = [other_col_names[i] for i in unused_indices]
            other_cols = DataHandler.select_columns(input, other_col_names)
            other_dims = DataHandler.get_shape(other_cols)
            if len(other_dims) == 1:
                other_dims = (other_dims[0], 1)
                other_cols = DataHandler.to_frame(other_cols, "__input__")
            if len(out_dims) == 1:
                out_dims = (out_dims[0], 1)
                out = DataHandler.to_frame(out, "__output__")
            if one_to_one_features:
                other_inds = unused_indices
                out_inds = indices
            else:
                other_inds = list(range(other_dims[1]))
                out_inds = list(range(other_dims[1], other_dims[1] + out_dims[1]))

            if other_dims[1] > out_dims[1]:
                out = DataHandler.concat([other_cols, out], axis=1)
                inds = list(np.argsort(other_inds + out_inds))
            else:
                out = DataHandler.concat([out, other_cols], axis=1)
                inds = list(np.argsort(out_inds + other_inds))

            out = DataHandler.select_columns(out, inds)
        return out

    def _make_columns_exclusive(self, columns):
        new_set = columns.copy()
        for i in reversed(range(0, len(columns) - 1)):
            if columns[i] is not None and columns[i + 1] is not None:
                new_set[i] = list(set(columns[i]) - set(columns[i + 1]))
        return new_set

    def _get_columns(
        self,
        X,
        *args,
        input_columns=None,
        input_feature_types=None,
        unused_columns=None,
        unused_feature_types=None,
        raise_if_missing=True,
    ):
        assert X is not None, "Input data is None"
        first_arg_row_num = DataHandler.get_shape(X)[0]
        assert first_arg_row_num > 0, "Input data has no rows"
        assert input_columns is None or isinstance(input_columns, list), (
            f"input_columns must be a list of column names or indices, "
            f"but got {type(input_columns)}"
        )

        def get_columns(
            X,
            input_columns=None,
            unused_columns=None,
            generate_cols=False,
            raise_if_missing=True,
        ):
            if X is None:
                return None, None, None
            col_names = DataHandler.get_column_names(X, generate_cols=True)
            col_names_set = set(col_names)
            if input_columns:
                if (
                    isinstance(input_columns, tuple)
                    and isinstance(input_columns[0], type)
                ) or isinstance(input_columns, type):
                    feature_type = input_columns
                    if isinstance(feature_type, type):
                        feature_type = (feature_type,)

                    try:
                        input_columns = DataHandler.get_column_names_by_feature_type(
                            X, feature_type=feature_type
                        )
                        if len(input_columns) == 0:
                            input_columns = None
                    except ValueError:
                        if generate_cols:
                            input_columns = DataHandler.get_column_names(
                                X, generate_cols=True
                            )
                        else:
                            input_columns = None

                else:
                    if isinstance(input_columns, (str, int)):
                        input_columns = [input_columns]
                    if not isinstance(input_columns[0], int):
                        missing_columns = set(input_columns) - col_names_set
                        if missing_columns and raise_if_missing:
                            raise ValueError(
                                f"Columns {missing_columns} not found in input dataset"
                            )
                        else:
                            input_columns = [
                                c for c in input_columns if c in col_names_set
                            ]
            if input_columns is None and unused_columns:
                if isinstance(unused_columns, (str, int)):
                    unused_columns = [unused_columns]
                if isinstance(unused_columns, tuple) or isinstance(
                    unused_columns, type
                ):
                    feature_type = unused_columns
                    if isinstance(feature_type, type):
                        feature_type = (feature_type,)
                    if is_datasets_available():
                        from datasets.features.features import _FEATURE_TYPES
                    else:
                        _FEATURE_TYPES = {}
                    if all(f in _FEATURE_TYPES.values() for f in feature_type):
                        try:
                            unused_columns = (
                                DataHandler.get_column_names_by_feature_type(
                                    X, feature_type=feature_type
                                )
                                or []
                            )
                        except ValueError:
                            if generate_cols:
                                unused_columns = []
                            else:
                                unused_columns = None
                if unused_columns is not None:
                    unused_columns = set(unused_columns)
                    input_columns = [c for c in col_names if c not in unused_columns]

            if (input_columns is None or len(input_columns) == 0) and generate_cols:
                input_columns = DataHandler.get_column_names(X, generate_cols=True)

            if input_columns:
                if isinstance(input_columns[0], str):
                    input_indices = DataHandler.get_column_indices(X, input_columns)
                else:
                    input_indices = input_columns
                    if DataHandler.supports_named_columns(get_data_format(X)):
                        cols = DataHandler.get_column_names(X, generate_cols=True)
                        input_columns = [cols[idx] for idx in input_indices]
                    else:
                        input_columns = None
                unused_indices = list(
                    sorted(set(range(len(col_names))) - set(input_indices))
                )
            else:
                return None, None, None

            return input_columns, input_indices, unused_indices

        arity = 1
        if input_columns and isinstance(input_columns, list):
            arity = len(input_columns)
        else:
            return None, None, None, None, None, None, None

        def parse_inputs(i):
            _input_columns = input_columns[i]

            _input_feature_types = None
            if input_feature_types is not None:
                _input_feature_types = input_feature_types[i]

            _unused_columns = None
            if unused_columns is not None:
                _unused_columns = unused_columns[i]

            _unused_feature_types = None
            if unused_feature_types is not None:
                _unused_feature_types = unused_feature_types[i]

            return (
                _input_columns or None,
                _input_feature_types or None,
                _unused_columns or None,
                _unused_feature_types or None,
            )

        _input_columns, _input_feature_types, _unused_columns, _unused_feature_types = (
            parse_inputs(0)
        )

        feature_names_in, feature_idx_in, unused_idx_in = get_columns(
            X,
            input_columns=_input_columns or _input_feature_types,
            unused_columns=_unused_columns or _unused_feature_types,
            generate_cols=True,
            raise_if_missing=raise_if_missing,
        )
        if _input_columns is not None and feature_idx_in is None:
            raise ValueError(
                f"Columns {_input_columns} not found in {DataHandler.get_column_names(X)}"
            )
        extra_names_in = None
        extra_idx_in = None
        unused_extra_idx_in = None
        offsets = None
        assert (
            arity == len(args) + 1
        ), f"Number of column sets ({arity}) must match the arity ({len(args) + 1})"
        if arity > 1 or len(args):
            extra_names_in = []
            extra_idx_in = []
            offsets = []
            if len(args) and not all(arg is None for arg in args):
                unused_extra_idx_in = []
                x_dims = DataHandler.get_shape(X)
                if len(x_dims) == 1:
                    offset = 1
                else:
                    offset = x_dims[1]
                for i, arg in enumerate(args):
                    (
                        _input_columns,
                        _input_feature_types,
                        _unused_columns,
                        _unused_feature_types,
                    ) = parse_inputs(i + 1)

                    if not is_bioset(X) and not is_dataset(X):
                        _unused_feature_types = None
                        _input_feature_types = None
                    if arg is None:
                        # look into the first input
                        _input_columns = _input_columns or _input_feature_types
                        _unused_columns = _unused_columns or _unused_feature_types
                        if _input_columns is None and _unused_columns is None:
                            _extra_names_in, _extra_idx_in, _unused_extra_idx_in = (
                                None,
                                None,
                                None,
                            )
                        else:
                            _extra_names_in, _extra_idx_in, _unused_extra_idx_in = (
                                get_columns(
                                    X,
                                    input_columns=_input_columns
                                    or _input_feature_types,
                                    unused_columns=_unused_columns
                                    or _unused_feature_types,
                                    generate_cols=True,
                                    raise_if_missing=raise_if_missing,
                                )
                            )
                        offsets.append(0)
                    else:
                        arg_dim = DataHandler.get_shape(arg)
                        _extra_names_in, _extra_idx_in, _unused_extra_idx_in = (
                            get_columns(
                                arg,
                                input_columns=_input_columns or _input_feature_types,
                                unused_columns=_unused_columns or _unused_feature_types,
                                generate_cols=True,
                                raise_if_missing=raise_if_missing,
                            )
                        )
                        arg_dim = DataHandler.get_shape(arg)

                        # only offset when the tables can be combined
                        if first_arg_row_num == arg_dim[0]:
                            offsets.append(offset)
                            if len(arg_dim) == 1:
                                offset += 1
                            else:
                                offset += arg_dim[1]
                    extra_names_in.append(_extra_names_in)
                    extra_idx_in.append(_extra_idx_in)
                    unused_extra_idx_in.append(_unused_extra_idx_in)

            else:
                for i in range(1, arity):
                    (
                        _input_columns,
                        _input_feature_types,
                        _unused_columns,
                        _unused_feature_types,
                    ) = parse_inputs(i)
                    _extra_names_in, _extra_idx_in, _unused_extra_idx_in = get_columns(
                        X,
                        input_columns=_input_columns or _input_feature_types,
                        unused_columns=_unused_columns or _unused_feature_types,
                        generate_cols=False,
                        raise_if_missing=raise_if_missing,
                    )
                    extra_names_in.append(_extra_names_in)
                    extra_idx_in.append(_extra_idx_in)
                    _extra_idx_in = set(_extra_idx_in or [])
                    unused_idx_in = [
                        idx for idx in unused_idx_in if idx not in _extra_idx_in
                    ]
                    offsets.append(0)
        return (
            feature_names_in,
            feature_idx_in,
            unused_idx_in,
            extra_names_in,
            extra_idx_in,
            unused_extra_idx_in,
            offsets,
        )

    def generate_fingerprint(self, fingerprint, config: BaseConfig):
        hash = Hasher()
        hash.update(fingerprint)
        hash_str = f"{self.__module__}.{self.__class__.__name__}"
        if hasattr(config, "version"):
            hash_str += f"@{config.version}"
        hash.update(hash_str)
        fingerprint = hash.hexdigest()
        fingerprint = fingerprint_from_kwargs(fingerprint, config.get_params())

        return fingerprint


class BaseProcessor(TransformationMixin):
    """
    Configures and manages data processing operations, supporting transformations and
    handling of various configurations and states, primarily designed for batch
    processing of data with options for multiprocessing and caching.

    Attributes:
        update_fingerprint (bool):
            Flag to determine if the fingerprint should be updated after processing.
        output_dtype (type): Data type for the output features.
        config (ProcessorConfig): Configuration object specifying processor settings.
        cache_files (list): List of cache file paths used during processing.

    Raises:
        ValueError: If an unsupported configuration type is provided."
    """

    # process attributes
    output_dtype = None

    # config attributes
    _config_class = ProcessorConfig
    config: ProcessorConfig = None

    # internal attributes for transformation
    _feature_dependent = True
    _input_columns = None
    _method_prefix: str = "_transform"
    _fingerprint = None
    _is_multiprocessing = False
    extra_names_in_ = []
    _selected_indices = None
    _unused_indices = None
    _extra_indices = None
    _unused_extra_indices = None

    # automatically generated attributes
    cache_files = None

    def __init__(self, config: Optional[ProcessorConfig] = None, **kwargs):
        add_new_attr = kwargs.pop("add_new_attr", False)
        ignore_none = kwargs.pop("ignore_none", False)

        if config is None:
            if hasattr(self, "_config_class"):
                self.config = self._config_class.from_dict(
                    kwargs, ignore_none=ignore_none, add_new_attr=add_new_attr
                )
        elif isinstance(config, ProcessorConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = self._config_class.from_dict(
                config, ignore_none=ignore_none, add_new_attr=add_new_attr
            )
        else:
            raise ValueError(f"Unsupported config type {type(config)}")
        if config is None:
            self = self.set_params(**kwargs)
        if kwargs.get("_function", None):
            self._function = kwargs["_function"]

    @classmethod
    def _from_config(cls, config: ProcessorConfig, **kwargs):
        """Instantiates the processor from a configuration object."""
        return cls(config=config, **kwargs)

    def __call__(self, batch: Union[pd.DataFrame, Dict[str, np.ndarray]], **kwargs):
        self = self._process_batches(batch, **kwargs)
        return batch

    @sync_backup_config
    def set_params(self, **kwargs):
        """Sets the parameters of the processor"""
        self.config = self.config.replace_defaults(**kwargs)
        return self

    @property
    def is_fitted(self):
        """bool: Whether the processor has been fitted."""
        return self.config.is_fitted

    @property
    def has_fit(self):
        """bool: Whether a fit function is found for the processor"""
        return self._get_method(_ORDERED_FORMATS, func_type="_fit") or self._get_method(
            _ORDERED_FORMATS, func_type="_fit", prefix=self.config._batch_method_prefix
        )

    @property
    def fingerprint(self):
        """str: The fingerprint of the processor."""
        return self._parse_fingerprint(self._fingerprint)

    def _parse_fingerprint(self, fingerprint):
        """Parses the fingerprint and returns the base fingerprint and the processor suffix."""
        base_fingerprint = fingerprint
        processor_suffix = f"-{self.config.processor_name}"
        if self.config.processor_type:
            processor_suffix += f"-{self.config.processor_type}"
        if self.config.experiment_name:
            processor_suffix += f"-{self.config.experiment_name}"
        return f"{base_fingerprint}{processor_suffix}"

    def _reset(self, config: ProcessorConfig):
        """Resets the processor to its initial state."""
        # reinstatiate the processor
        self._fingerprint = None
        self.__init__(config=config)

    @classmethod
    def from_config(
        cls, path_or_config: Union[str, os.PathLike, dict, "BaseConfig"], **kwargs
    ) -> T_PROC:
        """Instantiates the processor from a configuration file or object."""
        config = cls._config_class.from_config(path_or_config, add_new_attr=True)
        return cls(config=config, **kwargs)

    @classmethod
    def from_config_transform(cls, X, path: str, **kwargs):
        """Transforms the input data using the processor configuration."""
        self = cls.from_config(path, **kwargs)
        self.config.is_fitted = True
        return self.transform(X)

    def populate_map_kwargs(self, map_kwargs, cache_output, cache_file_name, num_proc):
        """
        Populates the map_kwargs with default values if not provided.

        Args:
            map_kwargs (dict): The keyword arguments for the map function.
            cache_output (bool): Whether to keep the processed data in memory.
            cache_file_name (str): The name of the cache file.
            num_proc (int): The number of processes to use for multiprocessing.

        Returns:
            dict: The updated map_kwargs.
        """
        if "keep_in_memory" not in map_kwargs:
            map_kwargs["keep_in_memory"] = not cache_output

        if "cache_file_name" not in map_kwargs:
            map_kwargs["cache_file_name"] = cache_file_name

        if "num_proc" not in map_kwargs:
            map_kwargs["num_proc"] = num_proc

        return map_kwargs

    def _validate_fit_params(self, arity):
        if self.config._input_columns is not None:
            if self.config._fit_input_feature_types is not None and len(
                self.config._input_columns
            ) != len(self.config._fit_input_feature_types):
                example_arg = ", ".join(["None" for _ in range(arity)])
                assert False, (
                    "`_fit_input_feature_types` is defined in "
                    f"{self.config.__class__.__name__} but does not match the arity of "
                    f"the fit function in {self.__class__.__name__} (i.e. len("
                    "self.config._fit_input_feature_types) != "
                    "len(self.config._input_columns) -> "
                    f"{len(self.config._fit_input_feature_types)} != "
                    f"{len(self.config._input_columns)}).\n"
                    "This can be corrected by doing, for example:\n"
                    f"_fit_input_feature_types = field(\n"
                    f"    default_factory=lambda: [{example_arg}], init=False, "
                    "repr=False\n"
                    ")"
                )
            if self.config._fit_unused_feature_types is not None and len(
                self.config._input_columns
            ) != len(self.config._fit_unused_feature_types):
                example_arg = ", ".join(["None" for _ in range(arity)])
                assert False, (
                    "`_fit_unused_feature_types` is defined in "
                    f"{self.config.__class__.__name__} but does not match the arity of "
                    f"the fit function in {self.__class__.__name__} (i.e. len("
                    "self.config._fit_unused_feature_types) != "
                    "len(self.config._input_columns) -> "
                    f"{len(self.config._fit_unused_feature_types)} != "
                    f"{len(self.config._input_columns)}).\n"
                    "This can be corrected by doing, for example:\n"
                    f"_fit_unused_feature_types = field(\n"
                    f"    default_factory=lambda: [{example_arg}], init=False, "
                    "repr=False\n"
                    ")"
                )

    def fit(
        self,
        X,
        raise_if_missing: bool = None,
        cache_output: bool = None,
        load_from_cache_file: bool = None,
        batched: bool = True,
        batch_size: int = 1000,
        batch_format: str = None,
        num_proc: int = None,
        map_kwargs: dict = {"fn_kwargs": {}},
        cache_dir: str = None,
        cache_file_name: str = None,
        fingerprint: str = None,
    ):
        """Must be implemented by subclasses if the processor is trainable."""
        if self.has_fit:
            # has a concrete fit method
            raise NotImplementedError("Processor must implement base `fit` method")
        return self._process_fit(
            X,
            raise_if_missing=raise_if_missing,
            cache_output=cache_output,
            load_from_cache_file=load_from_cache_file,
            batched=batched,
            batch_size=batch_size,
            batch_format=batch_format,
            num_proc=num_proc,
            map_kwargs=map_kwargs,
            cache_dir=cache_dir,
            cache_file_name=cache_file_name,
            fingerprint=fingerprint,
        )

    def _process_fit(
        self,
        X,
        *args,
        raise_if_missing: bool = None,
        cache_output: bool = None,
        load_from_cache_file: bool = None,
        batched: bool = True,
        batch_size: int = 1000,
        batch_format: str = None,
        num_proc: int = None,
        map_kwargs: dict = {"fn_kwargs": {}},
        cache_dir: str = None,
        cache_file_name: str = None,
        fingerprint: str = None,
    ) -> T_PROC:
        """
        Fits the processor to the input data, preparing it for transformation. This process
        may involve learning parameters from the data, validating input formats, and setting
        up caching mechanisms.

        Args:
            X (Union[np.ndarray, pd.DataFrame, Bioset, IterableDataset, DatasetDict, IterableDatasetDict]):
                The input data to fit the processor on. Can be a variety of types including
                numpy arrays, pandas DataFrames, or Hugging Face's `datasets` objects.
            *args (Any, optional):
                Additional input data to fit the processor on, such as target data for
                supervised learning tasks. Defaults to None.
            batched (bool, optional):
                Whether to process the data in batches. This can be beneficial for large datasets. Defaults to None,
                which will use the processor's default behavior.
            batch_transform (bool, optional):
                Specifies if transformation should be applied in batches. Defaults to None.
            batch_fit (bool, optional):
                Specifies if fitting should be applied in batches. Defaults to None.
            batch_size (int, optional):
                Size of each batch when `batched` is True. Defaults to 1000.
            map_kwargs (dict, optional):
                Additional keyword arguments to pass to the map function when processing datasets. Defaults to None.
            cache_output (bool, optional):
                Whether to keep the processed data in memory. Useful for avoiding disk IO. Defaults to None.
            batch_format (str, optional):
                The format to convert the input data to before processing. Defaults to None.
            batch_format_kwargs (dict, optional):
                Additional keyword arguments for the input format conversion. Defaults to None.
            fn_output_format_kwargs (dict, optional):
                Additional keyword arguments for the output format conversion. Defaults to None.
            split (str, optional):
                If the input is a DatasetDict or IterableDatasetDict, this specifies the split to process. Defaults to 'train'.
            cache_dir (str, Path, optional):
                Directory where processed datasets should be cached. Defaults to None, which uses the processor's default cache directory.
            fingerprint (str, optional):
                A unique identifier for the processing operation, used for caching. Defaults to None.
            load_from_cache_file (bool, optional):
                Whether to load the fitted processor from a cache file if available. Defaults to None, which follows the processor's default behavior.
            update_fingerprint (bool, optional):
                Whether to update the fingerprint after processing. Useful for ensuring uniqueness in caching. Defaults to None.
            keep_unused_columns (bool, optional):
                Whether to retain features in the dataset that are not processed by this processor. Defaults to True.

        Returns:
            self: The fitted processor instance.

        Raises:
            ValueError: If `input_columns` are specified but do not exist in the dataset.
        """

        funcs = self._get_method(_ORDERED_FORMATS, func_type="_fit")
        assert self.config._input_columns is not None or len(funcs) == 0, (
            f"The `fit` method of `{self.__class__.__name__}` must call:\n"
            "```\n"
            "self.config._input_columns = self._set_input_columns_and_arity(*args)"
            "\n```\n"
            "Where `*args` are the columns for each input dataset."
        )

        if not hasattr(self, "config_"):
            self.config_ = copy.deepcopy(self.config)
        else:
            self._reset(copy.deepcopy(self.config_))

        if is_dataset_dict(X):
            raise ValueError(
                "Please provide the dataset directly instead of the dictionary: processor.fit(dataset['train'])"
            )

        if cache_output is None:
            cache_output = self.config.enable_caching and self.config.cache_output

        if cache_output:
            self.config._data_fingerprint = getattr(
                X, "_fingerprint", None
            ) or fingerprint_from_data(X)
        else:
            self.config._data_fingerprint = None

        if cache_output:
            self._fingerprint = fingerprint

            if not fingerprint:
                self._fingerprint = fingerprint_from_kwargs(
                    self.config._data_fingerprint,
                    self.config.get_params(),
                )

            if cache_dir is not None:
                cache_dir = expand_path(str(cache_dir))
                cache_dir = os.path.join(cache_dir, "processors")

            cache_dir = generate_cache_dir(
                self,
                self.config._data_fingerprint,
                root_dir=cache_dir or biofit.config.BIOFIT_PROCESSORS_CACHE,
            )

            if cache_dir:
                if cache_file_name:
                    if is_remote_url(cache_file_name):
                        raise ValueError(
                            "`cache_file_name` is a remote URL. Please provide the "
                            "file name only. You can specify the directory using "
                            "`cache_dir`."
                        )
                    elif os.path.isabs(cache_file_name):
                        raise ValueError(
                            "`cache_file_name` is an absolute path. Please provide the "
                            "file name only. You can specify the directory using "
                            "`cache_dir`."
                        )
                self.cache_files = [
                    {
                        "filename": get_cache_file_name(
                            cache_dir, self.fingerprint, cache_file_name
                        )
                    }
                ]

        self._validate_fit_params(len(args) + 1)
        return self._fit(
            X,
            *args,
            funcs=funcs,
            cache_file_name=cache_file_name,
            input_columns=self.config._input_columns,
            raise_if_missing=raise_if_missing,
            cache_output=cache_output,
            load_from_cache_file=load_from_cache_file,
            cache_dir=cache_dir,
            batched=batched,
            batch_size=batch_size,
            batch_format=batch_format,
            map_kwargs=map_kwargs,
            num_proc=num_proc,
        )

    @keep_dataset_fingerprint
    def _fit(
        self,
        X,
        *args,
        funcs: List[Callable] = None,
        cache_file_name: str = None,
        input_columns: List[str] = None,
        raise_if_missing: bool = None,
        cache_output: bool = False,
        load_from_cache_file: bool = None,
        cache_dir: str = None,
        batched: bool = True,
        batch_size: int = 1000,
        batch_format: str = None,
        map_kwargs: dict = None,
        num_proc: int = None,
    ):
        """
        Fits the processor to the input data.

        Args:
            X (Any): The input data.
            y (Any, optional): The target data. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The fitted processor.
        """

        # upadate fit_config with dataset and processor info

        if cache_output is None:
            cache_output = self.config.enable_caching and self.config.cache_output

        if load_from_cache_file is None:
            load_from_cache_file = (
                self.config.enable_caching and self.config.load_from_cache_file
            )
        try:
            if load_from_cache_file and self.cache_files:
                cache_file_name = self.cache_files[0]["filename"]
                self = self.load_processed_estimator_from_cache(cache_file_name)
                logger.info(f"Loading cached processed estimator at {cache_file_name}")
            else:
                raise NonExistentCacheError
        except NonExistentCacheError:
            (
                selected_columns,
                selected_indices,
                unused_indices,
                extra_columns,
                extra_indices,
                unused_extra_indices,
                offsets,
            ) = self._get_columns(
                X,
                *args,
                input_columns=input_columns,
                input_feature_types=self.config._fit_input_feature_types,
                unused_columns=self.config.unused_columns,
                unused_feature_types=self.config._fit_unused_feature_types,
                raise_if_missing=raise_if_missing
                if raise_if_missing is not None
                else True,
            )

            self.config.feature_idx_in_ = selected_indices
            self.config.feature_names_in_ = selected_columns
            self.config.extra_names_in_ = extra_columns
            self.config.extra_idx_in_ = extra_indices

            extra = []
            extra_to_pass = []
            # two options: either all args are the same number of rows as X and we can
            # combine them or they are not and we need to pass them separately.
            # Advantage of combining is that we can apply multiprocessing or batching
            # by using the same indices. Also interops with HF's Bioset.map
            if len(args) and not all(arg is None for arg in args):
                main_dims = DataHandler.get_shape(X)
                combine_all = True
                for arg in args:
                    if arg is not None:
                        arg_dims = DataHandler.get_shape(arg)
                        if arg_dims[0] != main_dims[0]:
                            combine_all = False
                            break
                for i, arg in enumerate(args):
                    if DataHandler.supports_named_columns(arg) and combine_all:
                        cols = DataHandler.get_column_names(arg, generate_cols=True)
                        cols = [f"{c}_{i}" for c in cols]
                        extra.append(DataHandler.set_column_names(arg, cols))
                    else:
                        extra.append(arg)
                if combine_all:
                    input = DataHandler.concat(
                        [X] + [ext for ext in extra if ext is not None], axis=1
                    )
                    extra_to_pass = None
                else:
                    extra_to_pass = extra
                    extra = None
            else:
                input = X

            fit_map_kwargs, pooler = self._prepare_fit_kwargs(
                funcs,
                input,
                X,
                extra_inputs=extra,
                extra_untouched_inputs=extra_to_pass,
                selected_indices=selected_indices,
                unused_indices=unused_indices,
                extra_indices=extra_indices,
                unused_extra_indices=unused_extra_indices,
                offsets=offsets,
                map_kwargs=map_kwargs,
                num_proc=num_proc,
                batch_format=batch_format,
                batched=batched,
                batch_size=batch_size,
            )
            input, fit_map_kwargs = self._process_fit_input(input, **fit_map_kwargs)
            runner = None
            out = self
            if fit_map_kwargs["fn_kwargs"]["fn"]:
                if is_ray_available() and "ray" in sys.modules:
                    import ray.data

                    if isinstance(X, ray.data.Bioset):
                        fit_ray_kwargs = self._convert_map_kwargs_to_ray_kwargs(
                            fit_map_kwargs, batch_format=batch_format, is_fit=True
                        )
                        return X, fit_ray_kwargs

                pooler = pooler if pooler is not None else self._pool_fit

                @wraps(BaseProcessor.map)
                def runner(*args, **map_kwargs):
                    out = self.map(*args, **map_kwargs)
                    if len(out) == 1:
                        return out[0]
                    return pooler(out)

                if is_iterable_dataset(input):
                    from datasets import IterableDataset

                    @wraps(IterableDataset.map)
                    def runner(*args, **map_kwargs):
                        if len(args) > 0:
                            ds = args[0]
                            args = args[1:]
                        else:
                            ds = map_kwargs.pop("self")
                        return ds.map(*args, **map_kwargs)

                out = self.run(input, runner=runner, **fit_map_kwargs)

            self = self._process_fit_output(input, out)

            self.config.n_features_in_ = self.config.n_features_in_
            if (
                self.config.feature_idx_in_ is not None
                and self.config.feature_names_in_ is None
            ):
                self.config.n_features_in_ = len(self.config.feature_idx_in_)

            self.config._n_features_out = (
                self.config._n_features_out
                or self.config._n_features_out
                or getattr(self, "n_features_out", None)
            )

            temp_file = None
            if cache_output and self.cache_files:
                cache_file_name = (
                    Path(self.cache_files[0]["filename"]).resolve().as_posix()
                )
                cache_dir = os.path.dirname(cache_file_name)
                temp_file = tempfile.NamedTemporaryFile(
                    "wb", dir=cache_dir, delete=False
                )
                try:
                    self.config.save_to_cache(
                        temp_file.name, fingerprint=self.fingerprint
                    )
                except (Exception, KeyboardInterrupt):
                    temp_file.close()
                    if os.path.exists(temp_file.name):
                        os.remove(temp_file.name)
                    raise

                if temp_file and self.cache_files:
                    move_temp_file(temp_file, cache_file_name)
        self.config.is_fitted = True
        return self

    def _validate_transform_params(self, arity):
        """Validates the input arguments for the transform function.
        This is used to ensure that the input columns match the expected arity of the
        transform function. Arity is defined by the length of self._input_columns.
        """
        if self._input_columns is not None:
            if self.config._transform_input_feature_types is not None and len(
                self._input_columns
            ) != len(self.config._transform_input_feature_types):
                example_arg = ", ".join(["None" for _ in range(arity)])
                assert False, (
                    "`_transform_input_feature_types` is defined in "
                    f"{self.config.__class__.__name__} but does not match the arity of "
                    f"the transform function in {self.__class__.__name__} (i.e. len("
                    "self.config._transform_input_feature_types) != "
                    "len(self._input_columns) -> "
                    f"{len(self.config._transform_input_feature_types)} != "
                    f"{len(self._input_columns)}).\n"
                    "This can be corrected by doing, for example:\n"
                    f"_transform_input_feature_types = field(\n"
                    f"    default_factory=lambda: [{example_arg}], init=False, "
                    "repr=False\n"
                    ")"
                )
            if self.config._transform_unused_feature_types is not None and len(
                self._input_columns
            ) != len(self.config._transform_unused_feature_types):
                example_arg = ", ".join(["None" for _ in range(arity)])
                assert False, (
                    "`_transform_unused_feature_types` is defined in "
                    f"{self.config.__class__.__name__} but does not match the arity of "
                    f"the transform function in {self.__class__.__name__} (i.e. len("
                    "self.config._transform_unused_feature_types) != "
                    "len(self._input_columns) -> "
                    f"{len(self.config._transform_unused_feature_types)} != "
                    f"{len(self._input_columns)}).\n"
                    "This can be corrected by doing, for example:\n"
                    f"_transform_unused_feature_types = field(\n"
                    f"    default_factory=lambda: [{example_arg}], init=False, "
                    "repr=False\n"
                    ")"
                )

    def _process_transform(
        self,
        X,
        *args,
        keep_unused_columns: bool = None,
        raise_if_missing: bool = None,
        cache_output: bool = None,
        cache_dir: str = None,
        cache_file_name: str = None,
        load_from_cache_file: bool = None,
        batched: bool = None,
        batch_size: int = None,
        batch_format: str = None,
        output_format: str = None,
        map_kwargs: dict = None,
        num_proc: int = None,
        fingerprint: str = None,
    ):
        method_type = self._method_prefix[1:]

        assert self._input_columns is not None, (
            f"The `{method_type}` method of `{self.__class__.__name__}` must call:\n"
            "```\n"
            "self._input_columns = self._set_input_columns_and_arity(*args)"
            "\n```\n"
            "Where `*args` are the columns for each input dataset."
        )

        if is_dataset_dict(X):
            raise ValueError(
                "Please provide the dataset directly instead of the dictionary: processor.transform(dataset['train'])"
            )
        data_fingerprint = getattr(X, "_fingerprint", None) or fingerprint_from_data(X)
        cache_output = (
            cache_output is not None
            and cache_output
            or (self.config.enable_caching and self.config.cache_output)
        )

        if cache_output:
            if not fingerprint:
                hash = Hasher()
                hash.update(self._fingerprint)
                hash.update(data_fingerprint)
                fingerprint = fingerprint_from_kwargs(
                    hash.hexdigest(),
                    {
                        "input_columns": self._input_columns,
                        "unused_columns": self.config.unused_columns,
                    },
                )

            if cache_dir is not None:
                cache_dir = expand_path(str(cache_dir))
                cache_dir = os.path.join(cache_dir, "datasets")
            cache_dir = cache_dir or biofit.config.BIOFIT_DATASETS_CACHE
            cache_dir = generate_cache_dir(
                X,
                data_fingerprint,
                root_dir=cache_dir,
            )

        if cache_file_name:
            if is_remote_url(cache_file_name):
                raise ValueError(
                    "`cache_file_name` is a remote URL. Please provide the "
                    "file name only. You can specify the directory using "
                    "`cache_dir`."
                )
            elif os.path.isabs(cache_file_name):
                raise ValueError(
                    "`cache_file_name` is an absolute path. Please provide the "
                    "file name only. You can specify the directory using "
                    "`cache_dir`."
                )

        keep_unused_columns = (
            keep_unused_columns
            if keep_unused_columns is not None
            else self.config.keep_unused_columns
        )
        self.keep_unused_columns = keep_unused_columns
        if (
            self._input_columns is None
            and self._feature_dependent
            and data_fingerprint == self.config._data_fingerprint
        ):
            if DataHandler.supports_named_columns(X):
                cols = self.config.feature_names_in_ or self.config.feature_idx_in_
            else:
                cols = self.config.feature_idx_in_
            if (
                cols
                and self.config.extra_idx_in_
                and len(args) > 0
                and len(args) == len(self.config.extra_idx_in_)
            ):
                cols = [cols]
                for i in range(len(self.config.extra_idx_in_)):
                    if self.config.extra_names_in_[
                        i
                    ] and DataHandler.supports_named_columns(args[i]):
                        cols.append(self.config.extra_names_in_[i])
                    else:
                        cols.append(self.config.extra_idx_in_[i])

                self._input_columns = cols

        self._validate_transform_params(len(args) + 1)
        return self._transform(
            X,
            *args,
            input_columns=self._input_columns,
            keep_unused_columns=keep_unused_columns,
            raise_if_missing=raise_if_missing,
            cache_output=cache_output,
            cache_file_name=cache_file_name,
            cache_dir=cache_dir,
            load_from_cache_file=load_from_cache_file,
            batched=batched,
            batch_size=batch_size,
            batch_format=batch_format,
            output_format=output_format,
            map_kwargs=map_kwargs,
            num_proc=num_proc,
            fingerprint=fingerprint,
        )

    @keep_dataset_fingerprint
    def _transform(
        self,
        X,
        *args,
        input_columns: List[str] = None,
        keep_unused_columns: bool = None,
        raise_if_missing: bool = None,
        cache_output: bool = None,
        cache_file_name: str = None,
        cache_dir: str = None,
        load_from_cache_file: bool = None,
        batched: bool = None,
        batch_size: int = None,
        batch_format: str = None,
        output_format: str = None,
        map_kwargs: dict = None,
        num_proc: int = None,
        fingerprint: str = None,
    ):
        """
        Transforms the input data.

        Args:
            X (Any): The input data.
            y (Any, optional): The target data. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The computed processor.
        """
        if not self.is_fitted:
            raise NotFittedError

        (
            _,
            self._selected_indices,
            self._unused_indices,
            _,
            self._extra_indices,
            self._unused_extra_indices,
            offsets,
        ) = self._get_columns(
            X,
            *args,
            input_columns=input_columns,
            input_feature_types=self.config._transform_input_feature_types,
            unused_columns=self.config.unused_columns,
            unused_feature_types=self.config._transform_unused_feature_types,
            raise_if_missing=raise_if_missing if raise_if_missing is not None else True,
        )

        if len(args) and not all(arg is None for arg in args):
            extra = []
            for i, arg in enumerate(args):
                if arg is not None:
                    if DataHandler.supports_named_columns(arg):
                        cols = DataHandler.get_column_names(arg, generate_cols=True)
                        cols = [f"{c}_{i}" for c in cols]
                        extra.append(DataHandler.set_column_names(arg, cols))
                    else:
                        extra.append(arg)
            input = DataHandler.concat([X] + extra, axis=1)
        else:
            input = X

        if load_from_cache_file is None:
            load_from_cache_file = (
                self.config.enable_caching and self.config.load_from_cache_file
            )

        trans_map_kwargs = self._prepare_transform_kwargs(
            input,
            X,
            *args,
            selected_indices=self._selected_indices,
            unused_indices=self._unused_indices,
            extra_indices=self._extra_indices,
            unused_extra_indices=self._unused_extra_indices,
            offsets=offsets,
            cache_dir=cache_dir,
            new_fingerprint=fingerprint,
            map_kwargs=map_kwargs or {"fn_kwargs": {}},
            batch_format=batch_format,
            batched=batched,
            batch_size=batch_size,
            cache_output=cache_output,
            cache_file_name=cache_file_name,
            load_from_cache_file=load_from_cache_file,
            num_proc=num_proc,
            keep_unused_columns=keep_unused_columns,
        )

        out = None
        in_memory_table = None
        if trans_map_kwargs["fn_kwargs"]["fn"]:

            @wraps(BaseProcessor.map)
            def runner(*args, **map_kwargs):
                return DataHandler.concat(self.map(*args, **map_kwargs))

            if is_bioset(X) or is_dataset(X, iterable=False):
                if is_biosets_available():
                    from biosets import Bioset

                    wrap_map = Bioset.map
                else:
                    from datasets import Dataset

                    wrap_map = Dataset.map

                @wraps(wrap_map)
                def runner(*args, **map_kwargs):
                    if len(args) > 0:
                        ds = args[0]
                        args = args[1:]
                    else:
                        ds = map_kwargs.pop("self")
                    return ds.map(*args, **map_kwargs)

            elif is_iterable_dataset(X):
                from datasets import IterableDataset

                @wraps(IterableDataset.map)
                def runner(*args, **map_kwargs):
                    if len(args) > 0:
                        ds = args[0]
                        args = args[1:]
                    else:
                        ds = map_kwargs.pop("self")
                    return ds.map(*args, **map_kwargs)

            out = self.run(input, runner=runner, **trans_map_kwargs)

        out = self._process_transform_output(
            out,
            X,
            *args,
            output_format=output_format,
            keep_unused_columns=keep_unused_columns,
            selected_indices=self._selected_indices,
            unused_indices=self._unused_indices,
            fingerprint=trans_map_kwargs["new_fingerprint"],
        )

        if in_memory_table:
            X._data = in_memory_table

        return out

    def _process_extra_inds(
        self, orig_input, extra_inputs, extra_indices, unused_extra_indices
    ):
        """
        Processes extra indices for additional inputs.

        This method adjusts the indices for extra inputs by adding an offset based on the
        dimensions of the original input and the extra inputs. It ensures that the indices
        are correctly aligned with the combined input dimensions.

        Args:
            orig_input: The original input data.
            extra_inputs: A list of additional input data.
            extra_indices: A list of indices corresponding to the extra inputs.
            unused_extra_indices: A list of unused indices corresponding to the extra inputs.

        Returns:
            A tuple containing:
                - extra_inds: A list of adjusted indices for the extra inputs.
                - unused_extra_inds: A list of adjusted unused indices for the extra inputs.
        """
        assert not extra_inputs or extra_indices is not None, (
            "`extra_indices` was returned as `None` from "
            f"`{self.__class__.__name__}`. "
            f"Was `{self.__class__.__name__}._input_columns` or "
            f"`{self.__class__.__name__}.config._input_columns` set correctly?"
        )
        extra_inds = copy.deepcopy(extra_indices)
        unused_extra_inds = copy.deepcopy(unused_extra_indices)
        if extra_inputs and not all(arg is None for arg in extra_inputs):
            x_dims = DataHandler.get_shape(orig_input)
            if len(x_dims) == 1:
                x_dims = (x_dims[0], 1)
            offset = x_dims[1]
            extra_inds = []
            unused_extra_inds = []
            if unused_extra_indices is None:
                unused_extra_indices = [None] * len(extra_indices)
            for inds, un_inds, arg in zip(
                extra_indices, unused_extra_indices, extra_inputs
            ):
                if arg is not None:
                    if inds is not None and len(inds) > 0:
                        extra_inds.append([i + offset for i in inds])
                    else:
                        extra_inds.append(None)
                    if un_inds is not None and len(un_inds) > 0:
                        unused_extra_inds.append([i + offset for i in un_inds])
                    else:
                        unused_extra_inds.append(None)

                    arg_dims = DataHandler.get_shape(arg)
                    if len(arg_dims) == 1:
                        arg_dims = (arg_dims[0], 1)
                    offset += arg_dims[1]
                else:
                    extra_inds.append(None)
                    unused_extra_inds.append(None)
        return extra_inds, unused_extra_inds

    def _prepare_fit_kwargs(
        self,
        funcs,
        combined_inputs,
        orig_input,
        extra_inputs,
        extra_untouched_inputs,
        selected_indices=None,
        unused_indices=None,
        extra_indices=None,
        unused_extra_indices=None,
        offsets=None,
        map_kwargs={"fn_kwargs": {}},
        batch_format=None,
        batched=None,
        batch_size=None,
        num_proc=None,
    ):
        original_format = get_data_format(combined_inputs)

        poolers = None
        if (
            batch_size is not None
            and batch_size < DataHandler.get_shape(combined_inputs)[0]
            and funcs
        ):
            batchable_funcs = self._get_method(
                _ORDERED_FORMATS,
                func_type="_fit",
                prefix=self.config._batch_method_prefix,
            )
            if not batchable_funcs:
                batch_size = DataHandler.get_shape(combined_inputs)[0]
                if batched is not None:
                    logger.warning_once(
                        f"There are no batched fit functions available for {self.__class__.__name__}. "
                        "Using non-batched fit functions."
                    )
                batched = True
                batch_size = None
                batched = True
                poolers = None
            else:
                poolers = self._get_method(_ORDERED_FORMATS, func_type="_pool_fit")
                funcs = batchable_funcs
        if original_format == "ray" and batch_format is None:
            batch_format = ["pandas", "pd", "numpy", "np"]

        func, batch_format = self._get_target_func(funcs, original_format, batch_format)
        pooler = None
        if poolers:
            pooler, _ = self._get_target_func(poolers, original_format, batch_format)

        func_args = inspect.getfullargspec(func).args if func else []

        with_indices = (
            "indices" in func_args
            or "indexes" in func_args
            or "index" in func_args
            or "ind" in func_args
            or "inds" in func_args
            or "idx" in func_args
            or "i" in func_args
        )

        with_rank = "rank" in func_args or "rnk" in func_args or "r" in func_args
        map_kwargs = map_kwargs or {}
        map_kwargs = copy.deepcopy(map_kwargs)
        map_kwargs["with_indices"] = with_indices
        map_kwargs["with_rank"] = with_rank

        if "num_proc" not in map_kwargs:
            map_kwargs["num_proc"] = num_proc

        if (
            func
            and "num_proc" in map_kwargs
            and not func.__name__.startswith(self.config._batch_method_prefix)
        ):
            # cannot use num_proc with non-batched fit functions
            map_kwargs.pop("num_proc")

        map_kwargs["desc"] = self.config._fit_process_desc
        if batched is None or batched:
            map_kwargs["batched"] = True
            map_kwargs["batch_size"] = batch_size
        else:
            map_kwargs["batched"] = False
            map_kwargs["batch_size"] = 1

        extra_inds, unused_extra_inds = self._process_extra_inds(
            orig_input=orig_input,
            extra_inputs=extra_inputs,
            extra_indices=extra_indices,
            unused_extra_indices=unused_extra_indices,
        )

        fn_kwargs = {
            "fn": func,
            "func_type": "_fit",
            "extra_untouched_inputs": extra_untouched_inputs,
            "selected_indices": selected_indices,
            "unused_indices": unused_indices,
            "extra_indices": extra_inds,
            "unused_extra_indices": unused_extra_inds,
            "with_metadata": "metadata" in func_args,
            "in_format_kwargs": {
                "target_format": batch_format,
            },
            "out_format_kwargs": {
                "target_format": None,
            },
        }

        if "fn_kwargs" in map_kwargs:
            map_kwargs["fn_kwargs"].update(fn_kwargs)
        else:
            map_kwargs["fn_kwargs"] = fn_kwargs

        map_kwargs["new_fingerprint"] = self.fingerprint

        return map_kwargs, pooler

    def _prepare_transform_kwargs(
        self,
        combined_inputs,
        orig_input,
        *extra_inputs,
        selected_indices,
        unused_indices,
        extra_indices,
        unused_extra_indices,
        offsets=None,
        batch_format=None,
        map_kwargs={"fn_kwargs": {}},
        batched=None,
        batch_size=1000,
        cache_output=True,
        cache_file_name=None,
        cache_dir=None,
        load_from_cache_file=True,
        new_fingerprint=None,
        num_proc=None,
        keep_unused_columns=None,
    ):
        original_format = get_data_format(combined_inputs)
        input_format = batch_format

        funcs = self._get_method(_ORDERED_FORMATS, func_type=self._method_prefix)
        func, batch_format = self._get_target_func(funcs, original_format, input_format)

        map_kwargs = map_kwargs.copy()
        func_args = inspect.getfullargspec(func).args if func else []
        indices_args = [
            "indices",
            "indexes",
            "index",
            "ind",
            "inds",
            "idx",
            "i",
        ]
        rank_args = ["rank", "rnk", "r"]
        with_indices = any(arg in func_args for arg in indices_args)
        with_rank = any(arg in func_args for arg in rank_args)
        map_kwargs = copy.deepcopy(map_kwargs)
        map_kwargs["with_indices"] = with_indices
        map_kwargs["with_rank"] = with_rank

        map_kwargs["desc"] = getattr(
            self.config,
            self._method_prefix + "_process_desc",
            "Transforming data",
        )

        if "keep_in_memory" not in map_kwargs:
            map_kwargs["keep_in_memory"] = not cache_output

        if "cache_file_name" not in map_kwargs:
            map_kwargs["cache_file_name"] = cache_file_name

        if "num_proc" not in map_kwargs:
            map_kwargs["num_proc"] = num_proc

        if batched is None or batched:
            map_kwargs["batched"] = True
            map_kwargs["batch_size"] = batch_size
        elif batched:
            map_kwargs["batched"] = True
            map_kwargs["batch_size"] = batch_size
        else:
            map_kwargs["batched"] = False

        if "load_from_cache_file" not in map_kwargs:
            map_kwargs["load_from_cache_file"] = load_from_cache_file

        output_format = original_format
        if batch_format in _ARROW_WRITEABLE_FORMATS:
            output_format = batch_format
        elif original_format not in _ARROW_WRITEABLE_FORMATS:
            output_format = "arrow"

        map_kwargs["new_fingerprint"] = new_fingerprint

        features_out = None

        extra_inds, unused_extra_inds = self._process_extra_inds(
            orig_input=orig_input,
            extra_inputs=extra_inputs,
            extra_indices=extra_indices,
            unused_extra_indices=unused_extra_indices,
        )

        fn_kwargs = {
            "fn": func,
            "func_type": self._method_prefix,
            "with_metadata": "metadata" in func_args,
            "selected_indices": selected_indices,
            "unused_indices": unused_indices,
            "extra_indices": extra_inds,
            "unused_extra_indices": unused_extra_inds,
            "keep_unused_columns": keep_unused_columns,
            "in_format_kwargs": {
                "target_format": batch_format,
            },
            "out_format_kwargs": {
                "target_format": output_format,
            },
        }

        if "fn_kwargs" in map_kwargs:
            map_kwargs["fn_kwargs"].update(fn_kwargs)
        else:
            map_kwargs["fn_kwargs"] = fn_kwargs

        combined_inputs, map_kwargs = self._process_transform_input(
            combined_inputs, **map_kwargs
        )

        if hasattr(orig_input, "cache_files") and orig_input.cache_files:
            if cache_file_name is None:
                cache_files = orig_input.cache_files[0]["filename"]
                cache_dir = os.path.dirname(cache_files)
                cache_file_name = f"cache-{new_fingerprint}.arrow"
            cache_file_name = os.path.join(cache_dir, cache_file_name)

        if not load_from_cache_file or not (
            cache_file_name and os.path.exists(cache_file_name)
        ):
            if "features" not in map_kwargs or map_kwargs["features"] is None:
                unsel_inds = unused_indices if unused_indices else []
                if extra_inds:
                    unsel_inds += [
                        i
                        for sub in [
                            inds if inds is not None else [] for inds in extra_inds
                        ]
                        for i in sub
                    ]
                if unused_extra_inds:
                    unsel_inds += [
                        i
                        for sub in [
                            inds if inds is not None else []
                            for inds in unused_extra_inds
                        ]
                        for i in sub
                    ]
                unsel_inds = sorted(unsel_inds)
                features_out = self._get_features_out(
                    combined_inputs,
                    selected_indices=copy.deepcopy(selected_indices),
                    unselected_indices=unsel_inds,
                    one_to_one_features=self.config.one_to_one_features,
                    n_features_out=self.config._n_features_out,
                    keep_unused_columns=keep_unused_columns,
                )
                map_kwargs["features"] = features_out
            else:
                features_out = map_kwargs["features"]

            map_kwargs["fn_kwargs"]["feature_names"] = list(features_out.keys())
            map_kwargs["features"] = features_out
        return map_kwargs

    def fit_transform(
        self,
        X,
        *args,
        **kwargs,
    ):
        raise NotImplementedError(
            "The `fit_transform` method is not implemented for this processor."
        )

    def transform(self, X, *args, **kwargs):
        raise NotImplementedError(
            "The `transform` method is not implemented for this processor."
        )

    def _process_transform_input(self, X, **kwargs):
        return X, kwargs

    def _process_transform_output(self, output, input, *args, **kwargs):
        output_format = kwargs.get("output_format", None) or get_data_format(input)
        if output_format:
            if output is None:
                _method_prefix = self._method_prefix
                if self._method_prefix.startswith("_"):
                    _method_prefix = _method_prefix[1:]

                raise ValueError(
                    f"The output format is specified as `{output_format}` but the "
                    f"output from the {_method_prefix} method "
                    "is `None`."
                )
            output = DataHandler.to_format(output, output_format)
        if DataHandler.get_shape(output)[0] != DataHandler.get_shape(input)[0]:
            return output
        return output

    def get_params(self, deep=True, show_init_only=True, show_repr_only=True):
        """Get the parameters of the processor."""
        return self.config.get_params(
            deep=deep, show_init_only=show_init_only, show_repr_only=show_repr_only
        )

    def load_processed_estimator_from_cache(
        self, cache_file_name: Optional[Union[Path, str]] = None, **kwargs
    ):
        """Load a processed estimator from cache if it exists, otherwise throw an error."""
        # Check if we've already cached this computation (indexed by a hash)
        if cache_file_name and os.path.exists(cache_file_name):
            if isinstance(cache_file_name, Path):
                cache_file_name = cache_file_name.resolve().as_posix()
            self.config = self._config_class.from_config_file(cache_file_name)
            return self
        else:
            raise NonExistentCacheError

    def _get_features_out(
        self,
        X,
        selected_indices=None,
        unselected_indices=None,
        one_to_one_features=True,
        n_features_out=None,
        keep_unused_columns=False,
    ) -> "Features":
        features_out = None
        if unselected_indices is not None:
            unsel_inds = set(unselected_indices)
        else:
            unsel_inds = set()
        cols = DataHandler.get_column_names(X, generate_cols=True)
        assert cols is not None, "Could not generate column names from input data"
        if one_to_one_features:
            if selected_indices is not None:
                sel_inds = set(selected_indices)
            else:
                sel_inds = set(range(len(cols)))
        else:
            sel_inds = set()

        if is_bioset(X) or is_dataset(X):
            # get the output features, as well as the features that need to be reinserted
            features = X._info.features.copy()
        elif is_datasets_available():
            from datasets.features import Value

            features = {k: Value(dtype=v) for k, v in DataHandler.get_dtypes(X).items()}
        else:
            features = {k: None for k in cols}

        if keep_unused_columns:
            sel_inds = sel_inds.union(unsel_inds)

        out_cols = self._get_feature_names_out(
            n_features_out=n_features_out,
            input_features=cols,
            useful_feature_inds=list(sorted(sel_inds)),
            one_to_one_features=one_to_one_features,
        )
        features_out = {}
        if one_to_one_features:
            pa_type = None
            if self.output_dtype:
                pa_type = string_to_arrow(self.output_dtype)
            pos = 0
            for i in range(len(cols)):
                if i in unsel_inds:
                    if not keep_unused_columns:
                        continue
                    features_out[out_cols[pos]] = features[cols[i]]
                    pos += 1
                else:
                    k = out_cols[pos]
                    pos += 1
                    v = features[cols[i]]
                    if pa_type and hasattr(v, "dtype") and hasattr(v, "pa_type"):
                        setattr(v, "dtype", self.output_dtype)
                        setattr(v, "pa_type", pa_type)
                    features_out[k] = v
        else:
            if keep_unused_columns:
                features_out.update({cols[i]: features[cols[i]] for i in unsel_inds})
                # the transformed features are always appended to the end
                out_cols = out_cols[len(unsel_inds) :]
            if is_datasets_available():
                from datasets.features import Value

                if self.output_dtype:
                    features_out.update(
                        {k: Value(dtype=self.output_dtype) for k in out_cols}
                    )
                else:
                    dtypes = list(
                        set([features[cols[i]].dtype for i in selected_indices])
                    )
                    dtype = determine_upcast(dtypes)
                    features_out.update({k: Value(dtype=dtype) for k in out_cols})
            else:
                features_out.update({k: None for k in out_cols})

        if is_datasets_available():
            from datasets.features import Features

            features_out = Features(features_out)

        return features_out

    def _prepare_runner(self, X, **fn_kwargs):
        if fn_kwargs.get("with_metadata", False):
            if is_bioset(X) or is_dataset(X) and is_biosets_available():
                from biosets import get_feature_metadata

                feat_metadata = get_feature_metadata(X)
                feat_arrow_tbl = pa.Table.from_pylist(list(feat_metadata.values()))
                try:
                    feat_arrow_tbl = feat_arrow_tbl.add_column(
                        0, "features", pa.array(list(feat_metadata.keys()))
                    )
                    fn_kwargs["metadata"] = feat_arrow_tbl
                except Exception:
                    pass

        return fn_kwargs

    def run(
        self,
        X,
        runner: Optional[Callable] = None,
        fn_kwargs: dict = {},
        **map_kwargs,
    ):
        fn_kwargs = self._prepare_runner(X, **fn_kwargs)
        if runner:
            if is_bioset(X) or is_dataset(X):
                format = "arrow"
                if (
                    "in_format_kwargs" in fn_kwargs
                    and "target_format" in fn_kwargs["in_format_kwargs"]
                ):
                    format = fn_kwargs["in_format_kwargs"]["target_format"]
                    if format not in _ARROW_WRITEABLE_FORMATS:
                        format = "arrow"
                with X.formatted_as(format):
                    kwargs = get_kwargs({"fn_kwargs": fn_kwargs, **map_kwargs}, runner)
                    return runner(X, self._process_batches, **kwargs)
            else:
                kwargs = get_kwargs({"fn_kwargs": fn_kwargs, **map_kwargs}, runner)
                return runner(X, self._process_batches, **kwargs)
        else:
            return self._process_batches(X, **fn_kwargs)

    def _get_feature_names_out(
        self,
        input_features,
        n_features_out=0,
        useful_feature_inds=None,
        one_to_one_features=True,
    ):
        """
        Retrieves the feature names based on the input and output data.

        Args:
            input (Any): The input data.
            output (Any): The output data.

        Returns:
            list: The list of feature names.
        """
        out_features = [input_features[i] for i in useful_feature_inds]
        if one_to_one_features:
            if input_features is None:
                raise ValueError(
                    "Input features must be provided to generate output features"
                )
            if useful_feature_inds is None:
                useful_feature_inds = range(len(input_features))

            if self.config.features_out_suffix or self.config.features_out_prefix:
                if self.config.features_out_prefix:
                    out_features = [
                        f"{self.config.features_out_prefix}{i}" for i in out_features
                    ]

                if self.config.features_out_suffix:
                    out_features = [
                        f"{i}{self.config.features_out_suffix}" for i in out_features
                    ]
            return out_features
        else:
            _out_features = _generate_get_feature_names_out(self, n_features_out)
            if self.config.features_out_suffix or self.config.features_out_prefix:
                if self.config.features_out_prefix:
                    _out_features = [
                        f"{self.config.features_out_prefix}{col}"
                        for col in _out_features
                    ]
                if self.config.features_out_suffix:
                    _out_features = [
                        f"{col}{self.config.features_out_suffix}"
                        for col in enumerate(_out_features)
                    ]
                out_features.extend(_out_features)
                return out_features
            out_features.extend(_out_features)
            return out_features

    ## Functions for Datasets and IterableDatasets type ##

    def _convert_map_kwargs_to_ray_kwargs(
        self,
        map_kwargs: dict,
        batch_format=None,
        is_fit=True,
    ):
        ray_kwargs = None
        if is_fit:
            ray_kwargs = {
                "fn": self.__class__,
                "num_cpus": map_kwargs.get("num_proc", None),
                "num_gpus": map_kwargs.get("num_gpus", None),
                "batch_format": batch_format
                if batch_format in ["numpy", "pandas"]
                else None,
                "batch_size": map_kwargs.get("batch_size", None),
                "concurrency": 1,
                "fn_kwargs": map_kwargs["fn_kwargs"],
                "fn_constructor_args": (self.config,),
                "zero_copy_batch": True,
            }
        else:
            ray_kwargs = {
                "fn": self._process_batches,
                "num_cpus": map_kwargs.get("num_proc", None),
                "num_gpus": map_kwargs.get("num_gpus", None),
                "batch_format": batch_format
                if batch_format in ["numpy", "pandas"]
                else None,
                "batch_size": map_kwargs.get("batch_size", None),
                "fn_kwargs": map_kwargs["fn_kwargs"],
                "fn_constructor_args": (self.config,),
                "zero_copy_batch": False,
            }

        return ray_kwargs

    def _dummy_func(self, X, *args, **kwargs):
        """for creating fingerprint and returning the input data as is when no function is provided."""
        return X

    def _process_batches(self, X, *fn_args, **fn_kwargs):
        func = fn_kwargs.get("fn", None)
        if fn_kwargs["func_type"] == "_fit":
            input, _fn_args, _fn_kwargs = self._process_fit_batch_input(
                X, *fn_args, **fn_kwargs
            )
        else:
            input, _fn_args, _fn_kwargs = self._process_transform_batch_input(
                X, *fn_args, **fn_kwargs
            )

        out = func(input, *_fn_args, **_fn_kwargs)

        if isinstance(out, BaseProcessor):
            return self._process_fit_batch_output(out)
        return self._process_transform_batch_output(X, out, **fn_kwargs)

    def _validate_fit_func_args(self, func, *fn_args):
        required_args = get_required_args(func)
        noun_plural = "argument" if len(required_args) == 1 else "arguments"
        verb_tense = "was" if len(fn_args) == 0 else "were"
        assert len(required_args) == (len(fn_args) + 1), (
            f"`{self.__class__.__name__}.fit` requires {len(required_args)} "
            f"{noun_plural} {tuple(required_args)}, "
            f"but only {len(fn_args) + 1} {verb_tense} "
            "provided. Either provide the missing arguments or provide the input "
            f"columns found in '{required_args[0]}', if applicable."
        )

    def _process_fit_batch_input(self, X, *fn_args, **fn_kwargs):
        func = fn_kwargs.get("fn", None)
        assert X is not None, "No input data provided."
        assert func is not None, "No function provided for processing the data."
        in_format_kwargs = fn_kwargs.get("in_format_kwargs", {})
        selected_indices = fn_kwargs.get("selected_indices", [])
        in_format_kwargs["input_columns"] = selected_indices
        input = DataHandler.to_format(X, **in_format_kwargs)
        _fn_kwargs = get_kwargs(fn_kwargs, func)
        extra_indices = fn_kwargs.get("extra_indices", [])
        extra_untouched_inputs = fn_kwargs.get("extra_untouched_inputs", [])
        fn_args = list(fn_args)
        if extra_untouched_inputs:
            for arg in extra_untouched_inputs:
                in_format_kwargs["input_columns"] = None
                arg = DataHandler.to_format(arg, **in_format_kwargs)
                fn_args.append(arg)
        elif extra_indices is not None:
            for cols in extra_indices:
                if cols is not None and len(cols):
                    in_format_kwargs["input_columns"] = cols
                    arg = DataHandler.to_format(X, **in_format_kwargs)
                    fn_args.append(arg)

        self._validate_fit_func_args(func, *fn_args)
        return input, tuple(fn_args), _fn_kwargs

    def _process_fit_batch_output(self, out):
        return out

    def _process_transform_batch_input(self, X, *fn_args, **fn_kwargs):
        assert X is not None, "No input data was provided for processing."
        func = fn_kwargs.get("fn", None)
        in_format_kwargs = fn_kwargs.get("in_format_kwargs", {})
        selected_indices = fn_kwargs.get("selected_indices", [])
        in_format_kwargs["input_columns"] = selected_indices
        input = DataHandler.to_format(X, **in_format_kwargs)
        _fn_kwargs = get_kwargs(fn_kwargs, func)
        extra_indices = fn_kwargs.get("extra_indices", [])
        if extra_indices:
            fn_args = list(fn_args)
            for i, cols in enumerate(extra_indices):
                if cols is not None and len(cols):
                    in_format_kwargs["input_columns"] = cols
                    extra_arg = DataHandler.to_format(X, **in_format_kwargs)
                    fn_args.insert(i, extra_arg)
        return input, fn_args, _fn_kwargs

    def _process_transform_batch_output(self, input, output, **fn_kwargs):
        selected_indices = fn_kwargs.get("selected_indices", None)
        unused_indices = fn_kwargs.get("unused_indices", None)
        keep_unused_columns = fn_kwargs.get("keep_unused_columns", None)
        feature_names = fn_kwargs.get("feature_names", None)
        out_dims = DataHandler.get_shape(output)
        if len(out_dims) == 1:
            out_dims = (out_dims[0], 1)
            output = DataHandler.to_frame(output, "__output__")

        out_format_kwargs = fn_kwargs.get("out_format_kwargs", {})
        output = DataHandler.to_format(output, **out_format_kwargs)
        if keep_unused_columns:
            output = self._reinsert_columns(
                input,
                output,
                selected_indices,
                unused_indices,
                one_to_one_features=self.config.one_to_one_features,
            )
        if feature_names is not None and len(feature_names) > 0:
            output = DataHandler.set_column_names(output, feature_names)
        return output

    def _process_fit_input(self, input, **kwargs):
        return input, kwargs

    def _process_fit_output(self, input, out):
        return out

    def __repr__(self):
        from sklearn.utils._pprint import _EstimatorPrettyPrinter

        N_MAX_ELEMENTS_TO_SHOW = 30
        pp = _EstimatorPrettyPrinter(
            compact=True,
            indent=1,
            indent_at_name=True,
            n_max_elements_to_show=N_MAX_ELEMENTS_TO_SHOW,
        )
        config_str = pp.pformat(self.config)
        if config_str.startswith(self.config.__class__.__name__):
            config_str = config_str[len(self.config.__class__.__name__) :]
        repr_ = self.__class__.__name__ + config_str
        return repr_

    def _repr_mimebundle_(self, **kwargs):
        """Mime bundle used by jupyter kernels to display estimator"""
        from sklearn._config import get_config
        from sklearn.utils._estimator_html_repr import estimator_html_repr

        output = {"text/plain": repr(self)}
        if get_config()["display"] == "diagram":
            output["text/html"] = estimator_html_repr(self)
        return output

    def shard(self, X, num_shards, index, contiguous):
        if not 0 <= index < num_shards:
            raise ValueError("index should be in [0, num_shards-1]")
        num_rows = DataHandler.get_shape(X)[0]
        if contiguous:
            div = num_rows // num_shards
            mod = num_rows % num_shards
            start = div * index + min(index, mod)
            end = start + div + (1 if index < mod else 0)
            indices = range(start, end)
        else:
            indices = np.arange(index, num_rows, num_shards)

        return DataHandler.select_rows(X, indices)

    def _pool_fit(self, fitted_processors):
        """Pool the results of the map function."""
        out = fitted_processors[0]
        if len(fitted_processors) > 1:
            raise NotImplementedError(
                f"Pooling results from multiple processes is not supported for {self.__class__.__name__}"
            )
        return out

    def map(
        self,
        X,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        with_rank: bool = False,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        cache_output: bool = True,
        input_columns: Optional[list] = None,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        desc: Optional[str] = None,
    ):
        if batch_size is None:
            batch_size = DataHandler.get_shape(X)[0]
        num_shards = num_proc if num_proc is not None else 1
        if batched and drop_last_batch:
            pbar_total = (
                DataHandler.get_shape(X)[0]
                // num_shards
                // batch_size
                * num_shards
                * batch_size
            )
        else:
            pbar_total = DataHandler.get_shape(X)[0]

        if num_proc:
            if is_bioset(X) or is_dataset(X):
                shards = [
                    X.shard(
                        num_shards=num_shards,
                        index=rank,
                        contiguous=True,
                        keep_in_memory=not cache_output,
                    )
                    for rank in range(num_proc)
                ]
            else:
                shards = [
                    self.shard(
                        X,
                        num_shards=num_proc,
                        index=rank,
                        contiguous=True,
                    )
                    for rank in range(num_proc)
                ]
        else:
            shards = [X]

        dataset_kwargs = {
            "shard": X,
            "function": function,
            "with_indices": with_indices,
            "with_rank": with_rank,
            "batched": batched,
            "batch_size": batch_size,
            "drop_last_batch": drop_last_batch,
            "input_columns": input_columns,
            "fn_kwargs": fn_kwargs,
        }

        kwargs_per_job = [
            {
                **dataset_kwargs,
                "shard": shards[rank],
                "rank": rank,
                "offset": sum(len(s) for s in shards[:rank]),
            }
            for rank in range(num_shards)
        ]

        if len(kwargs_per_job) < num_shards:
            logger.info(
                f"Reprocessing {len(kwargs_per_job)}/{num_shards} shards because some of them were missing from the cache."
            )

        processed_data = [None] * num_shards
        shards_done = 0
        if num_proc is not None and num_proc > 1:
            with Pool(len(kwargs_per_job)) as pool:
                logger.info(f"Spawning {num_proc} processes")
                if is_datasets_available():
                    from datasets.utils import tqdm
                else:
                    from tqdm.auto import tqdm
                with tqdm(
                    unit=" examples",
                    total=pbar_total,
                    desc=(desc or "Map") + f" (num_proc={num_proc})",
                ) as pbar:
                    for rank, done, content in iflatmap_unordered(
                        pool,
                        BaseProcessor._map_single,
                        kwargs_iterable=kwargs_per_job,
                    ):
                        if done:
                            shards_done += 1
                            logger.debug(
                                f"Finished processing shard number {rank} of {num_shards}."
                            )
                            processed_data[rank] = content
                        else:
                            pbar.update(content)
        else:
            processed_data = None
            if is_datasets_available():
                from datasets.utils import tqdm
            else:
                from tqdm.auto import tqdm

            with tqdm(
                unit=" examples",
                total=pbar_total,
                desc=desc or "Map",
            ) as pbar:
                for rank, done, content in BaseProcessor._map_single(**dataset_kwargs):
                    if done:
                        shards_done += 1
                        logger.debug(
                            f"Finished processing shard number {rank} of {num_shards}."
                        )
                        processed_data = content
                    else:
                        pbar.update(content)
            assert processed_data is not None, "Failed to retrieve the result from map"

            return [processed_data]
        for kwargs in kwargs_per_job:
            del kwargs["shard"]
        return processed_data

    @staticmethod
    def _map_single(
        shard: Any,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        with_rank: bool = False,
        input_columns: Optional[List[str]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        fn_kwargs: Optional[dict] = None,
        rank: Optional[int] = None,
        offset: int = 0,
    ) -> Iterable[Tuple[int, bool, Union[int, Any]]]:
        """Apply a function to all the elements in the table (individually or in batches)
        and update the table (if function does update examples).

        Args:
            shard (`datasets.Bioset`): Bioset to map the transform on.
            function (`Callable`): with one of the following signature:
                - `function(example: Dict[str, Any]) -> Dict[str, Any]` if `batched=False` and `with_indices=False` and `with_rank=False`
                - `function(example: Dict[str, Any], *extra_args) -> Dict[str, Any]` if `batched=False` and `with_indices=True` and/or `with_rank=True` (one extra arg for each)
                - `function(batch: Dict[str, List]) -> Dict[str, List]` if `batched=True` and `with_indices=False` and `with_rank=False`
                - `function(batch: Dict[str, List], *extra_args) -> Dict[str, List]` if `batched=True` and `with_indices=True` and/or `with_rank=True` (one extra arg for each)

                For advanced usage, the function can also return a `pyarrow.Table`.
                Moreover if your function returns nothing (`None`), then `map` will run your function and return the dataset unchanged.
                If no function is provided, default to identity function: lambda x: x
            with_indices (`bool`, defaults to `False`): Provide example indices to `function`. Note that in this case the signature of `function` should be `def function(example, idx[, rank]): ...`.
            with_rank (`bool`, default `False`): Provide process rank to `function`. Note that in this case the signature of `function` should be `def function(example[, idx], rank): ...`.
            input_columns (`Optional[List[str]]`, defaults to `None`): The columns to be passed into `function` as
                positional arguments. If `None`, a dict mapping to all formatted columns is passed as one argument.
            batched (`bool`, defaults to `False`): Provide batch of examples to `function`
            batch_size (`int`, optional, defaults to `1000`): Number of examples per batch provided to `function` if `batched=True`
                `batch_size <= 0` or `batch_size == None`: Provide the full dataset as a single batch to `function`
            drop_last_batch (`bool`, default: `False`): Whether a last batch smaller than the batch_size should be
                dropped instead of being processed by the function.
            fn_kwargs (`Dict`, optional, defaults to `None`): Keyword arguments to be passed to `function`
            rank: (`int`, optional, defaults to `None`): If specified, this is the process rank when doing multiprocessing
            offset: (`int`, defaults to 0): If specified, this is an offset applied to the indices passed to `function` if `with_indices=True`.
        """
        if fn_kwargs is None:
            fn_kwargs = {}

        # If we do batch computation but no batch size is provided, default to the full dataset
        if batched and (batch_size is None or batch_size <= 0):
            batch_size = DataHandler.get_shape(shard)[0]

        # We set this variable to True after processing the first example/batch in
        # `apply_function_on_filtered_inputs` if the map function returns a dict.
        # If set to False, no new arrow table will be created

        update_data = None
        input_formatter = None
        if is_bioset(shard) or is_dataset(shard):
            from datasets.arrow_dataset import get_formatter

            format_kwargs = shard._format_kwargs.copy()
            # Lazy formatting is only available for the default format (None/python)
            if not input_columns and shard._format_type is None:
                format_kwargs["lazy"] = True
            input_formatter = get_formatter(
                shard._format_type,
                features=shard._info.features,
                **format_kwargs,
            )

        def apply_function_on_filtered_inputs(data, indices, offset=0):
            """Utility to apply the function on a selection of columns."""
            nonlocal update_data
            if (
                isinstance(data, pa.Table)
                and input_formatter
                and is_datasets_available()
            ):
                from datasets.arrow_dataset import format_table

                inputs = format_table(
                    data,
                    0 if not batched else range(data.num_rows),
                    format_columns=input_columns,
                    formatter=input_formatter,
                )
            else:
                inputs = data
            fn_args = (
                [inputs]
                if input_columns is None
                else [DataHandler.select_column(inputs, col) for col in input_columns]
            )
            if offset == 0:
                effective_indices = indices
            else:
                effective_indices = (
                    [i + offset for i in indices]
                    if isinstance(indices, list)
                    else indices + offset
                )
            additional_args = ()
            if with_indices:
                additional_args += (effective_indices,)
            if with_rank:
                additional_args += (rank,)
            processed_inputs = function(*fn_args, *additional_args, **fn_kwargs)
            return processed_inputs

        num_examples_progress_update = 0
        # Optionally initialize the writer as a context manager
        try:
            if is_bioset(shard) or is_dataset(shard, iterable=False):
                arrow_formatted_shard = shard.with_format("arrow")

                if not batched:
                    shard_iterable = enumerate(arrow_formatted_shard)
                else:
                    num_rows = (
                        DataHandler.get_shape(shard)[0]
                        if not drop_last_batch
                        else DataHandler.get_shape(shard)[0] // batch_size * batch_size
                    )
                    shard_iterable = zip(
                        range(0, num_rows, batch_size),
                        arrow_formatted_shard.iter(
                            batch_size, drop_last_batch=drop_last_batch
                        ),
                    )
            else:
                if not batched:
                    shard_iterable = enumerate(shard)
                else:
                    num_rows = (
                        DataHandler.get_shape(shard)[0]
                        if not drop_last_batch
                        else DataHandler.get_shape(shard)[0] // batch_size * batch_size
                    )
                    shard_iterable = zip(
                        range(0, num_rows, batch_size),
                        DataHandler.iter(
                            shard, batch_size, drop_last_batch=drop_last_batch
                        ),
                    )
            processors = None
            if not batched:
                _time = time.time()
                for i, example in shard_iterable:
                    processor = apply_function_on_filtered_inputs(
                        example, i, offset=offset
                    )
                    if isinstance(processor, BaseProcessor):
                        processors = processor
                    else:
                        processors = processors or []
                        processors.append(processor)

                    num_examples_progress_update += 1
                    if time.time() > _time + biofit.config.PBAR_REFRESH_TIME_INTERVAL:
                        _time = time.time()
                        yield rank, False, num_examples_progress_update
                        num_examples_progress_update = 0
            else:
                _time = time.time()
                for i, batch in shard_iterable:
                    num_examples_in_batch = DataHandler.get_shape(batch)[0]
                    indices = list(
                        range(
                            *(
                                slice(i, i + batch_size).indices(
                                    DataHandler.get_shape(shard)[0]
                                )
                            )
                        )
                    )  # Something simpler?
                    processor = apply_function_on_filtered_inputs(
                        batch,
                        indices,
                        offset=offset,
                    )
                    if isinstance(processor, BaseProcessor):
                        processors = processor
                    else:
                        processors = processors or []
                        processors.append(processor)
                    num_examples_progress_update += num_examples_in_batch
                    if time.time() > _time + biofit.config.PBAR_REFRESH_TIME_INTERVAL:
                        _time = time.time()
                        yield rank, False, num_examples_progress_update
                        num_examples_progress_update = 0

        except (Exception, KeyboardInterrupt):
            yield rank, False, num_examples_progress_update
            raise

        yield rank, False, num_examples_progress_update

        if isinstance(processors, BaseProcessor):
            yield rank, True, processors
        elif isinstance(processors, list):
            if len(processors) > 0:
                yield rank, True, DataHandler.concat(processors)
            else:
                yield rank, True, None
        else:
            yield rank, True, processors

    def cleanup_cache_files(self, X=None, cache_dir=None, cache_file_name=None) -> int:
        """Clean up cache files generated by the processor."""
        count = 0
        cache_files = []
        if not self.cache_files and X is not None:
            data_fingerprint = getattr(
                X, "_fingerprint", None
            ) or fingerprint_from_data(X)

            if cache_dir is not None:
                cache_dir = os.path.join(expand_path(str(cache_dir)), "processors")
            cache_dir = generate_cache_dir(X, data_fingerprint, cache_dir=cache_dir)
            if cache_dir:
                cache_files = [
                    {
                        "filename": get_cache_file_name(
                            cache_dir, self.fingerprint, cache_file_name
                        )
                    }
                ]
        else:
            cache_files = self.cache_files

        for cache_file in cache_files:
            if os.path.exists(cache_file["filename"]):
                os.remove(cache_file["filename"])
                count += 1
        return count
