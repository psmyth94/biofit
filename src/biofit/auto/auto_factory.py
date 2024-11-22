import importlib
from collections import OrderedDict
from typing import TYPE_CHECKING

from biocore.utils.import_util import is_transformers_available

from biofit.utils import logging

from ..auto.configuration_auto import (
    PROCESSOR_CATEGORY_MAPPING_NAMES,
    PROCESSOR_TYPE_MAPPING_NAMES,
    AutoConfig,
)

if TYPE_CHECKING:
    from biofit.processing import BaseProcessor

logger = logging.get_logger(__name__)


def _get_model_class(config, model_mapping: "_LazyAutoMapping"):
    supported_models = model_mapping[type(config)]
    if not isinstance(supported_models, (list, tuple)):
        return supported_models

    name_to_model = {model.__name__: model for model in supported_models}
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in name_to_model:
            return name_to_model[arch]
        elif f"TF{arch}" in name_to_model:
            return name_to_model[f"TF{arch}"]
        elif f"Flax{arch}" in name_to_model:
            return name_to_model[f"Flax{arch}"]

    # If not architecture is set in the config or match the supported models, the first element of the tuple is the
    # defaults.
    return supported_models[0]


def _get_class(config, model_mapping):
    supported_models = model_mapping[type(config)]
    return supported_models


class _BaseAutoProcessorClass:
    # Base class for auto preprocessors.
    _processor_mapping = None
    _experiment_mapping = None

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def for_processor(self, processor_name, **kwargs):
        config = AutoConfig.for_processor(processor_name, **kwargs)
        return self.from_config(config, **kwargs)

    @classmethod
    def from_config(cls, config, **kwargs):
        if type(config) in cls._processor_mapping.keys():
            preprocessor_class: "BaseProcessor" = _get_class(
                config, cls._processor_mapping
            )
            return preprocessor_class._from_config(config, **kwargs)

        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoProcessor: {cls.__name__}.\n"
            f"Processor type should be one of {', '.join(c.__name__ for c in cls._processor_mapping.keys())}."
        )

    @classmethod
    def register(cls, config_class, processor_class: "BaseProcessor", exist_ok=False):
        """
        Register a new model for this class.

        Args:
            _config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            model_class ([`PreTrainedModel`]):
                The model to register.
        """
        if hasattr(processor_class, "_config_class") and not issubclass(
            config_class, processor_class._config_class
        ):
            raise ValueError(
                "The model class you are passing has a `_config_class` attribute that is not consistent with the "
                f"config class you passed (model has {processor_class._config_class} and you passed {config_class}. Fix "
                "one of those so they match!"
            )
        cls._processor_mapping.register(
            config_class, processor_class, exist_ok=exist_ok
        )

    @classmethod
    def register_experiment(
        cls, experiment_name, config_classes, processor_classes, exist_ok=False
    ):
        """
        Register list of processors for an experiment.
        """
        if not isinstance(config_classes, (list, tuple)):
            config_classes = [config_classes]
        if not isinstance(processor_classes, (list, tuple)):
            processor_classes = [processor_classes]

        if len(config_classes) != len(processor_classes):
            raise ValueError(
                "config_classes and processor_classes should have the same length."
            )

        if experiment_name not in cls._experiment_mapping:
            cls._experiment_mapping[experiment_name] = _LazyAutoMapping(
                OrderedDict(), OrderedDict()
            )
        for config_class, processor_class in zip(config_classes, processor_classes):
            if hasattr(processor_class, "_config_class") and not issubclass(
                config_class, processor_class._config_class
            ):
                raise ValueError(
                    "The model class you are passing has a `_config_class` attribute that is not consistent with the "
                    f"config class you passed (model has {processor_class._config_class} and you passed {config_class}. Fix "
                    "one of those so they match!"
                )

            if (
                config_class in cls._experiment_mapping[experiment_name]
                and not exist_ok
            ):
                raise ValueError(
                    f"Configuration {config_class.__name__} is already used by a "
                    f"processor for experiment {experiment_name}."
                )
            cls._experiment_mapping[experiment_name].register(
                config_class, processor_class, exist_ok=exist_ok
            )

            if (
                processor_class in cls._experiment_mapping[experiment_name]
                and not exist_ok
            ):
                raise ValueError(
                    f"Processor {processor_class.__name__} is already used by a "
                    f"config for experiment {experiment_name}."
                )
            cls._experiment_mapping[experiment_name].register(
                processor_class, config_class, exist_ok=exist_ok
            )


class _BaseAutoModelClass:
    # Base class for auto models.
    _processor_mapping = None

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def for_model(cls, model_name, *model_args, **kwargs):
        config = AutoConfig.for_processor(model_name, **kwargs)
        return cls.from_config(config, *model_args, **kwargs)

    @classmethod
    def from_config(cls, config, **kwargs):
        if type(config) in cls._processor_mapping.keys():
            model_class: "BaseProcessor" = _get_model_class(
                config, cls._processor_mapping
            )
            return model_class._from_config(config, **kwargs)

        if is_transformers_available():
            logger.warning(
                "Could not find a matching model for this configuration. "
                "Searching for a model in the Transformers library instead."
            )

            from transformers.models.auto.auto_factory import (
                _BaseAutoModelClass as _HfBaseAutoModelClass,
            )

            return _HfBaseAutoModelClass.from_config(config, **kwargs)

        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._processor_mapping.keys())}."
        )

    @classmethod
    def from_pretrained(cls, pretrained_estimator_name_or_path, *model_args, **kwargs):
        if is_transformers_available():
            from transformers.models.auto.auto_factory import (
                _BaseAutoModelClass as _HfBaseAutoModelClass,
            )

            return _HfBaseAutoModelClass.from_pretrained(
                pretrained_estimator_name_or_path, *model_args, **kwargs
            )
        else:
            raise EnvironmentError(
                f"Using {cls.__name__}.from_pretrained requires the transformers library to be installed. "
                "You can install it with `pip install transformers`"
            )

    @classmethod
    def register(cls, config_class, processor_class: "BaseProcessor", exist_ok=False):
        """
        Register a new model for this class.

        Args:
            _config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            model_class ([`PreTrainedModel`]):
                The model to register.
        """
        if hasattr(processor_class, "_config_class") and issubclass(
            config_class, processor_class._config_class
        ):
            raise ValueError(
                "The model class you are passing has a `_config_class` attribute that is not consistent with the "
                f"config class you passed (model has {processor_class._config_class} and you passed {config_class}. Fix "
                "one of those so they match!"
            )
        cls._processor_mapping.register(
            config_class, processor_class, exist_ok=exist_ok
        )


def insert_head_doc(docstring, head_doc=""):
    if len(head_doc) > 0:
        return docstring.replace(
            "one of the model classes of the library ",
            f"one of the model classes of the library (with a {head_doc} head) ",
        )
    return docstring.replace(
        "one of the model classes of the library ",
        "one of the base model classes of the library ",
    )


def get_values(model_mapping):
    result = []
    for model in model_mapping.values():
        if isinstance(model, (list, tuple)):
            result += list(model)
        else:
            result.append(model)

    return result


def getattribute_from_module(module, attr):
    if attr is None:
        return None
    if isinstance(attr, tuple):
        return tuple(getattribute_from_module(module, a) for a in attr)
    if hasattr(module, attr):
        return getattr(module, attr)
    # Some of the mappings have entries estimator_type -> object of another model type. In that case we try to grab the
    # object at the top level.
    genomicsml_module = importlib.import_module("biofit")

    if module != genomicsml_module:
        try:
            return getattribute_from_module(genomicsml_module, attr)
        except ValueError:
            raise ValueError(
                f"Could not find {attr} neither in {module} nor in {genomicsml_module}!"
            )
    else:
        raise ValueError(f"Could not find {attr} in {genomicsml_module}!")


class _LazyAutoMapping(OrderedDict):
    """
    " A mapping config to object (model or tokenizer for instance) that will load keys and values when it is accessed.

    Args:
        - config_mapping: The map model type to config class
        - model_mapping: The map model type to model (or tokenizer) class
    """

    def __init__(self, config_mapping, processor_mapping):
        self._config_mapping = config_mapping
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
        self._processor_mapping = processor_mapping
        self._processor_mapping._processor_mapping = self
        self._extra_content = {}
        self._modules = {}

    def __len__(self):
        common_keys = set(self._config_mapping.keys()).intersection(
            self._processor_mapping.keys()
        )
        return len(common_keys) + len(self._extra_content)

    def __getitem__(self, key):
        if key in self._extra_content:
            return self._extra_content[key]
        processor_type = self._reverse_config_mapping[key.__name__]
        if processor_type in self._processor_mapping:
            processor_name = self._processor_mapping[processor_type]
            return self._load_attr_from_module(processor_type, processor_name)

        # Maybe there was several model types associated with this config.
        estimator_types = [
            k for k, v in self._config_mapping.items() if v == key.__name__
        ]
        for ptype in estimator_types:
            if ptype in self._processor_mapping:
                processor_name = self._processor_mapping[ptype]
                return self._load_attr_from_module(ptype, processor_name)
        raise KeyError(key)

    def _load_attr_from_module(self, module_name, attr):
        if module_name not in self._modules:
            processor_category = PROCESSOR_CATEGORY_MAPPING_NAMES.get(
                module_name, "models"
            )
            processor_type = PROCESSOR_TYPE_MAPPING_NAMES.get(module_name, None)
            if processor_type is not None:
                package_name = f"biofit.{processor_category}.{processor_type}"
            else:
                package_name = f"biofit.{processor_category}"
            self._modules[module_name] = importlib.import_module(
                f".{module_name}", package_name
            )
        return getattribute_from_module(self._modules[module_name], attr)

    def keys(self):
        mapping_keys = [
            self._load_attr_from_module(key, name)
            for key, name in self._config_mapping.items()
            if key in self._processor_mapping.keys()
        ]
        return mapping_keys + list(self._extra_content.keys())

    def get(self, key, default):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __bool__(self):
        return bool(self.keys())

    def values(self):
        mapping_values = [
            self._load_attr_from_module(key, name)
            for key, name in self._processor_mapping.items()
            if key in self._config_mapping.keys()
        ]
        return mapping_values + list(self._extra_content.values())

    def items(self):
        mapping_items = [
            (
                self._load_attr_from_module(key, self._config_mapping[key]),
                self._load_attr_from_module(key, self._processor_mapping[key]),
            )
            for key in self._processor_mapping.keys()
            if key in self._config_mapping.keys()
        ]
        return mapping_items + list(self._extra_content.items())

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, item):
        if item in self._extra_content:
            return True
        if (
            not hasattr(item, "__name__")
            or item.__name__ not in self._reverse_config_mapping
        ):
            return False
        estimator_type = self._reverse_config_mapping[item.__name__]
        return estimator_type in self._processor_mapping

    def register(self, key, value, exist_ok=False):
        """
        Register a new processor in this mapping.
        """
        if hasattr(key, "__name__") and key.__name__ in self._reverse_config_mapping:
            estimator_type = self._reverse_config_mapping[key.__name__]
            if estimator_type in self._processor_mapping.keys() and not exist_ok:
                raise ValueError(f"'{key}' is already used by a processor.")

        self._extra_content[key] = value


class _LazyAutoMapping(OrderedDict):
    """
    " A mapping config to object (model or tokenizer for instance) that will load keys and values when it is accessed.

    Args:
        - config_mapping: The map model type to config class
        - model_mapping: The map model type to model (or tokenizer) class
    """

    def __init__(self, config_mapping, processor_mapping):
        self._config_mapping = config_mapping
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
        self._processor_mapping = processor_mapping
        self._processor_mapping._processor_mapping = self
        self._extra_content = {}
        self._modules = {}

    def __len__(self):
        common_keys = set(self._config_mapping.keys()).intersection(
            self._processor_mapping.keys()
        )
        return len(common_keys) + len(self._extra_content)

    def __getitem__(self, key):
        if key in self._extra_content:
            return self._extra_content[key]
        processor_type = self._reverse_config_mapping[key.__name__]
        if processor_type in self._processor_mapping:
            processor_name = self._processor_mapping[processor_type]
            return self._load_attr_from_module(processor_type, processor_name)

        # Maybe there was several model types associated with this config.
        estimator_types = [
            k for k, v in self._config_mapping.items() if v == key.__name__
        ]
        for ptype in estimator_types:
            if ptype in self._processor_mapping:
                processor_name = self._processor_mapping[ptype]
                return self._load_attr_from_module(ptype, processor_name)
        raise KeyError(key)

    def _load_attr_from_module(self, module_name, attr):
        if module_name not in self._modules:
            processor_category = PROCESSOR_CATEGORY_MAPPING_NAMES.get(
                module_name, "models"
            )
            processor_type = PROCESSOR_TYPE_MAPPING_NAMES.get(module_name, None)
            if processor_type is not None:
                package_name = f"biofit.{processor_category}.{processor_type}"
            else:
                package_name = f"biofit.{processor_category}"
            self._modules[module_name] = importlib.import_module(
                f".{module_name}", package_name
            )
        return getattribute_from_module(self._modules[module_name], attr)

    def keys(self):
        mapping_keys = [
            self._load_attr_from_module(key, name)
            for key, name in self._config_mapping.items()
            if key in self._processor_mapping.keys()
        ]
        return mapping_keys + list(self._extra_content.keys())

    def get(self, key, default):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __bool__(self):
        return bool(self.keys())

    def values(self):
        mapping_values = [
            self._load_attr_from_module(key, name)
            for key, name in self._processor_mapping.items()
            if key in self._config_mapping.keys()
        ]
        return mapping_values + list(self._extra_content.values())

    def items(self):
        mapping_items = [
            (
                self._load_attr_from_module(key, self._config_mapping[key]),
                self._load_attr_from_module(key, self._processor_mapping[key]),
            )
            for key in self._processor_mapping.keys()
            if key in self._config_mapping.keys()
        ]
        return mapping_items + list(self._extra_content.items())

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, item):
        if item in self._extra_content:
            return True
        if (
            not hasattr(item, "__name__")
            or item.__name__ not in self._reverse_config_mapping
        ):
            return False
        estimator_type = self._reverse_config_mapping[item.__name__]
        return estimator_type in self._processor_mapping

    def register(self, key, value, exist_ok=False):
        """
        Register a new processor in this mapping.
        """
        if hasattr(key, "__name__") and key.__name__ in self._reverse_config_mapping:
            estimator_type = self._reverse_config_mapping[key.__name__]
            if estimator_type in self._processor_mapping.keys() and not exist_ok:
                raise ValueError(f"'{key}' is already used by a processor.")

        self._extra_content[key] = value
