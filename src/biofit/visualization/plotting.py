import inspect
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from biocore import DataHandler
from biocore.utils.import_util import is_ipywidgets_available, is_matplotlib_available
from biocore.utils.naming import camelcase_to_snakecase
from biocore.utils.py_util import is_dataset_dict

from biofit.integration.R.r_caller import RCaller
from biofit.processing import (
    _ORDERED_FORMATS,
    TransformationMixin,
    sync_backup_config,
)
from biofit.utils import (
    _build_cache_dir,
    fingerprint_from_data,
    has_ext,
    has_separator,
    logging,
    move_temp_file,
)
from biofit.visualization.plotting_utils import (
    display_image_carousel,
    is_in_notebook,
)

from ..processing import BaseConfig, SelectedColumnTypes, SelectedFeatureTypes

logger = logging.get_logger(__name__)


ORDERED_PLOTTER_FORMATS = _ORDERED_FORMATS + ["dataset", "ds"]


def _processor_info_from_fingerprint(fingerprint: str):
    if fingerprint is None:
        return "", "", ""
    processor_info = fingerprint.split("-")
    ds_name = ""
    if len(processor_info) == 5:
        _, processor_name, processor_type, ds_name, _ = processor_info
    elif len(processor_info) == 4:
        _, processor_name, processor_type, ds_name = processor_info
    elif len(processor_info) == 3:
        _, processor_name, processor_type = processor_info
    else:
        return "", "", ""
    return processor_name, processor_type, ds_name


@dataclass
class PlotterConfig(BaseConfig):
    path: str = field(default=None, init=True, repr=True)
    device: str = field(default="png", init=True, repr=False)
    fingerprint: str = field(default=None, init=True, repr=False)
    unused_columns: SelectedColumnTypes = field(default=None, init=True, repr=False)
    raise_if_missing: bool = field(default=True, init=True, repr=False)
    cache_dir: str = field(default=None, init=True, repr=False)
    version: str = field(default="0.0.0", init=True, repr=False)
    savefig_kwargs: dict = field(default_factory=dict, init=True, repr=False)

    _input_columns: SelectedColumnTypes = field(default=None, init=False, repr=False)
    _compare: bool = field(default=False, init=False, repr=False)
    _fit_input_feature_types: SelectedFeatureTypes = field(
        default=None, init=False, repr=False
    )
    _fit_unused_feature_types: SelectedFeatureTypes = field(
        default=None, init=False, repr=False
    )
    _transform_input_feature_types: SelectedFeatureTypes = field(
        default=None, init=False, repr=False
    )
    _transform_unused_feature_types: SelectedFeatureTypes = field(
        default=None, init=False, repr=False
    )
    r_code: str = field(default=None, init=False, repr=False)
    r_source: str = field(default=None, init=False, repr=False)
    main_method: str = field(default=None, init=False, repr=False)

    processor_type: str = field(default="", init=False, repr=False)
    processor_name: str = field(default="", init=False, repr=False)
    experiment_name: str = field(default="", init=False, repr=False)

    # automatically generated attributes

    feature_idx_in_: List[int] = field(default=None, init=False, repr=False)
    feature_names_in_: List[str] = field(default=None, init=False, repr=False)
    extra_idx_in_: List[List[int]] = field(default=None, init=False, repr=False)
    extra_names_in_: List[List[str]] = field(default=None, init=False, repr=False)


class BasePlotter(TransformationMixin):
    """All plotters should inherit from this class.

    Attributes:
        config (PlotterConfig, *optional*): The configuration for the plotter.
        r_caller (RCaller, *optional*): The RCaller object.
        plotter (function, *optional*): The R function to call.
    """

    r_caller: RCaller = None
    _config_class = PlotterConfig
    plotter: Optional[str] = None
    config: PlotterConfig

    def __init__(self, config: Optional[PlotterConfig] = None, **kwargs):
        add_new_attr = kwargs.pop("add_new_attr", False)
        ignore_none = kwargs.pop("ignore_none", False)

        if config is None:
            if hasattr(self, "_config_class"):
                self.config = self._config_class.from_dict(
                    kwargs, ignore_none=ignore_none, add_new_attr=add_new_attr
                )
        elif isinstance(config, PlotterConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = self._config_class.from_dict(
                config, ignore_none=ignore_none, add_new_attr=add_new_attr
            )
        else:
            raise ValueError(f"Unsupported config type {type(config)}")
        if config is None or len(kwargs) > 0:
            self = self.set_params(**kwargs)
        if (
            self.config.r_source is not None
            and self.config.main_method is not None
            and self.plotter is None
        ):
            self.r_caller = RCaller.from_script(self.config.r_source)
            self.r_caller.verify_r_dependencies(
                install_missing=kwargs.get("install_missing")
            )
            self.plotter = self.r_caller.get_method(self.config.main_method)
        if kwargs.get("_function", None):
            self._function = kwargs["_function"]

    @sync_backup_config
    def set_params(self, **kwargs):
        self.config = self.config.replace_defaults(**kwargs)
        if self.config.r_source and self.config.main_method:
            self.r_caller = RCaller.from_script(self.config.r_source)
            self.plotter = self.r_caller.get_method(self.config.main_method)

        return self

    @classmethod
    def _from_config(cls, config: PlotterConfig, **kwargs):
        return cls(config=config, **kwargs)

    def plot(self, X, *args, **kwargs):
        if is_dataset_dict(X):
            raise ValueError(
                "Cannot plot a DatasetDict or IterableDatasetDict. Please provide a Dataset or IterableDataset."
            )
        return self._plot(X, *args, **kwargs)

    def _plot(self, X, *args, **kwargs):
        """
        Transforms the input data.

        Args:
            X (Any): The input data.
            *args: Additional arguments. These are additional table or array-like data to be used in the plotting.
            **kwargs: Additional keyword arguments. These are objects or parameters to be used in the plotting.

        Returns:
            Any: The computed processor.
        """
        show = kwargs.pop("show", True)
        output_dir = kwargs.pop("output_dir", None)
        self.config, kwargs = self.config.replace_defaults(
            ignore_none=True, return_unused_kwargs=True, **kwargs
        )

        args = list(args)

        plot_funcs = self._get_method(ORDERED_PLOTTER_FORMATS, "plot")
        plot_func, target_format = self._get_target_func(
            plot_funcs,
            source_format=DataHandler.get_format(X),
            target_formats=kwargs.get("input_format", None),
            accepted_formats=ORDERED_PLOTTER_FORMATS,
        )

        fingerprint = kwargs.pop("fingerprint", None) or self.config.fingerprint

        fig = None
        if plot_func:
            (
                self.feature_names_in_,
                self.feature_idx_in_,
                _,
                self.extra_names_in_,
                self.extra_idx_in_,
                _,
                _,
            ) = self._get_columns(
                X,
                *args,
                input_columns=self.config._input_columns,
                unused_columns=self.config.unused_columns,
                input_feature_types=self.config._transform_input_feature_types,
                unused_feature_types=self.config._transform_unused_feature_types,
                raise_if_missing=self.config.raise_if_missing,
            )

            path_to_save = self.config.path

            # required args are ones without defaults
            required_args = [
                p.default == p.empty
                for p in inspect.signature(plot_func).parameters.values()
            ][1:]
            if (
                not self.config._compare
                and self.extra_idx_in_ is not None
                and len(self.extra_idx_in_) > 0
                and DataHandler.supports_named_columns(X)
            ):
                new_cols = self._make_columns_exclusive(
                    [self.feature_names_in_] + self.extra_names_in_
                )

                self.feature_names_in_ = new_cols[0]
                self.feature_idx_in_ = sorted(
                    DataHandler.get_column_indices(X, self.feature_names_in_)
                )

                self.extra_names_in_ = new_cols[1:]
                extra_idx_in_ = []
                for i, names in enumerate(self.extra_names_in_):
                    if len(args) and args[i] is not None:
                        extra_idx_in_.append(
                            sorted(
                                DataHandler.get_column_indices(
                                    args[i], names, raise_if_missing=False
                                )
                            )
                        )
                    elif names is not None:
                        extra_idx_in_.append(
                            sorted(
                                DataHandler.get_column_indices(
                                    X, names, raise_if_missing=True
                                )
                            )
                        )
                    else:
                        if required_args[i]:
                            raise ValueError(
                                f"Missing required argument {i} for plot function {plot_func}"
                            )
                        extra_idx_in_.append(self.extra_idx_in_[i])
                self.extra_idx_in_ = extra_idx_in_

            data_fingerprint = fingerprint_from_data(X)
            if fingerprint is None:
                fingerprint = self.generate_fingerprint(data_fingerprint, self.config)

            input, args, kwargs = self._process_plot_input(X, *args, **kwargs)
            args = list(args)
            if len(args) == 0:
                if self.extra_idx_in_:
                    for i, inds in enumerate(self.extra_idx_in_):
                        if inds is not None:
                            args.append(
                                DataHandler.to_format(
                                    X, target_format, input_columns=inds
                                )
                            )
                        else:
                            args.append(None)
            else:
                for i, (inds, arg) in enumerate(zip(self.extra_idx_in_, args)):
                    if arg is not None:
                        args[i] = DataHandler.to_format(
                            arg, target_format, input_columns=inds
                        )
                    elif inds is not None:
                        args[i] = DataHandler.to_format(
                            X, target_format, input_columns=inds
                        )

            input = DataHandler.to_format(
                input, target_format, input_columns=self.feature_idx_in_
            )
            temp_dir = tempfile.mkdtemp()
            temp_file = os.path.join(temp_dir, fingerprint)
            temp_dir = Path(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_dir = temp_dir.resolve().as_posix()
            self.config.path = temp_file + f".{self.config.device}"
            output_dir = None if output_dir is None else Path(output_dir)
            file_name = None
            if path_to_save:
                if has_separator(path_to_save):
                    if has_ext(path_to_save):
                        output_dir = (
                            Path(path_to_save).parent
                            if output_dir is None
                            else output_dir
                        )
                        file_name = Path(path_to_save).name
                    else:
                        output_dir = (
                            Path(path_to_save) if output_dir is None else output_dir
                        )
                        file_name = None
                elif has_ext(path_to_save):
                    file_name = str(path_to_save)

            if output_dir is None:
                if kwargs.get("processor", None) and kwargs["processor"].cache_files:
                    output_dir = (
                        Path(kwargs["processor"].cache_files[0]["filename"]).parent
                        / "plots"
                    )
                elif hasattr(X, "cache_files") and X.cache_files:
                    output_dir = Path(X.cache_files[0]["filename"]).parent / "plots"
                else:
                    output_dir = Path(_build_cache_dir(X, data_fingerprint)) / "plots"
            if file_name is None:
                cls_name = camelcase_to_snakecase(
                    self.__class__.__name__.replace("_plotter", "")
                )
                file_name = f"{fingerprint}_{cls_name}.{self.config.device}"

            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_dir = output_dir.resolve().as_posix()

            fig = plot_func(input, *args, **kwargs)
            if "matplotlib" in sys.modules:
                import matplotlib.pyplot as plt
                from matplotlib.artist import Artist

                if isinstance(fig, Artist):
                    if output_dir:
                        fig.figure.savefig(
                            os.path.join(output_dir, file_name),
                            format=self.config.device,
                            **self.config.savefig_kwargs,
                        )

            if output_dir:

                def get_or_move_images(images, output_dir, move_images=False):
                    image_paths = []
                    if len(images) > 1:
                        for fn in images:
                            old_name = fn.resolve().as_posix()
                            new_name = f"{output_dir}/{fn.name}"
                            if move_images:
                                image_paths.append(new_name)
                                move_temp_file(old_name, new_name)
                            else:
                                image_paths.append(old_name)
                        if move_images:
                            logger.info(f"Saved {len(images)} plots to {output_dir}")
                    elif len(images) == 1:
                        old_name = images[0].resolve().as_posix()
                        if move_images:
                            image_paths = f"{output_dir}/{file_name}"
                            move_temp_file(old_name, image_paths)
                            logger.info(f"Saved plot to {image_paths}")
                        else:
                            image_paths = old_name

                    return image_paths

                # move all files within temp_dir to output_dir
                images = [fn for fn in Path(temp_dir).glob(f"*.{self.config.device}")]
                image_paths = get_or_move_images(images, output_dir, move_images=True)

                _image_paths = []
                if self.config.device != "png":
                    _images = [fn for fn in Path(temp_dir).glob("*.png")]
                    _image_paths = get_or_move_images(
                        _images, output_dir, move_images=False
                    )
                if len(_image_paths) > 0:
                    image_paths = _image_paths
                    images = _images

                if len(image_paths) > 1:
                    if is_in_notebook() and show and is_ipywidgets_available():
                        display_image_carousel(image_paths, "png")
                elif len(image_paths) == 1:
                    if is_in_notebook() and show:
                        if (
                            is_matplotlib_available()
                            and "matplotlib" in sys.modules
                            and isinstance(fig, plt.Figure)
                        ):
                            # ignore warning about non-interactive backend
                            warnings.filterwarnings("ignore")
                            fig.show()
                            warnings.filterwarnings("default")
                        else:
                            from IPython.display import Image, display

                            display(
                                Image(
                                    image_paths,
                                    embed=True,
                                    format="png",
                                    width=720,
                                )
                            )
                    elif (
                        is_matplotlib_available()
                        and "matplotlib" in sys.modules
                        and isinstance(fig, plt.Figure)
                    ):
                        plt.close(fig)
                else:
                    logger.warning("No plots were generated")
        return fig

    def _process_plot_input(self, input, *args, **kwargs):
        return input, args, kwargs
