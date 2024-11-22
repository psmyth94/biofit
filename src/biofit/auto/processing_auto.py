from collections import OrderedDict
from typing import List

from biocore.utils.import_util import is_biosets_available
from biocore.utils.inspect import get_kwargs
from sklearn.pipeline import Pipeline

from biofit.auto.auto_factory import (
    _BaseAutoProcessorClass,
    _get_class,
    _LazyAutoMapping,
)
from biofit.processing import BaseProcessor

from .configuration_auto import (
    CONFIG_MAPPING_NAMES,
    EXPERIMENT_CONFIG_MAPPING_NAMES,
    PROCESSOR_MAPPING_NAMES,
    AutoPreprocessorConfig,
)

PROCESSOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, PROCESSOR_MAPPING_NAMES)
EXPERIMENT_MAPPING = {
    k: _LazyAutoMapping(
        v,
        OrderedDict(
            [
                (k2, PROCESSOR_MAPPING_NAMES[k2])
                for k2 in v
                if k2 in PROCESSOR_MAPPING_NAMES
            ]
        ),
    )
    for k, v in EXPERIMENT_CONFIG_MAPPING_NAMES.items()
}


class AutoProcessor(_BaseAutoProcessorClass):
    _processor_mapping = PROCESSOR_MAPPING


class ProcessorPipeline(Pipeline):
    """A pipeline of processors."""

    def __init__(self, processors: List[BaseProcessor] = None, **kwargs):
        if "steps" in kwargs:
            return super().__init__(**kwargs)

        self.processors = processors
        steps = []
        for i, processor in enumerate(processors):
            if not isinstance(processor, tuple):
                if hasattr(processor, "config"):
                    steps.append(
                        (
                            f"{processor.config.processor_name}_{i + 1}",
                            processor,
                        )
                    )
                else:
                    steps.append((f"{processor.__class__.__name__}_{i + 1}", processor))
            else:
                steps.append(processor)
        super().__init__(steps, **kwargs)

    def fit_transform(self, X, y=None, **kwargs):
        for processor in self.processors:
            if isinstance(processor, tuple):
                p = processor[-1]
            else:
                p = processor
            fit_transform_kwargs = get_kwargs(kwargs, p.fit_transform)
            X = p.fit_transform(X, y, **fit_transform_kwargs)
        return X

    def pop(self, index):
        self.steps.pop(index)
        return self.processors.pop(index)


class AutoPreprocessor(_BaseAutoProcessorClass):
    _processor_mapping = PROCESSOR_MAPPING
    _experiment_mapping = EXPERIMENT_MAPPING

    @classmethod
    def for_dataset(cls, dataset_name, **kwargs):
        """
        Create a preprocessor pipeline for a given dataset.

        Args:
            dataset: str
                The dataset name.
        Returns:
            ProcessorPipeline
                The preprocessor pipeline for the dataset.
        """
        if is_biosets_available():
            from biosets.packaged_modules import EXPERIMENT_TYPE_ALIAS
        else:
            EXPERIMENT_TYPE_ALIAS = {}

        dataset_name = EXPERIMENT_TYPE_ALIAS.get(dataset_name, dataset_name)
        _processor_mapping = cls._experiment_mapping[dataset_name]
        configs = AutoPreprocessorConfig.for_experiment(dataset_name, **kwargs)
        procs = [
            _get_class(config, _processor_mapping)._from_config(config)
            for config in configs
        ]

        processors = ProcessorPipeline(procs)
        return processors
