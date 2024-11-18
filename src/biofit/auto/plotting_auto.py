from typing import List, Union

from biocore.utils.import_util import is_biosets_available
from biocore.utils.inspect import get_kwargs

from biofit.auto.auto_factory import (
    _BaseAutoProcessorClass,
    _LazyAutoMapping,
)
from biofit.processing import BaseProcessor
from biofit.visualization.plotting import BasePlotter
from biofit.visualization.plotting_utils import (
    display_image_carousel,
    is_in_notebook,
)

from .configuration_auto import (
    DATASET_PLT_TO_MAPPER_NAMES,
    PLOTTER_CONFIG_MAPPING_NAMES,
    PLOTTER_MAPPING_NAMES,
    AutoPlotterConfig,
)
from .processing_auto import AutoPreprocessor, ProcessorPipeline

PLOTTER_MAPPING = _LazyAutoMapping(PLOTTER_CONFIG_MAPPING_NAMES, PLOTTER_MAPPING_NAMES)


class PlotterPipeline:
    def __init__(self, plotters: List[BasePlotter], processors: List[BaseProcessor]):
        self.plotters = plotters
        self.processors = processors

    def plot(self, X, *args, fit=True, **kwargs):
        from datasets import Dataset, IterableDataset

        from biofit import Bioset

        if not isinstance(X, (Bioset, Dataset, IterableDataset)):
            raise ValueError("X must be a Bioset or huggingface Dataset.")
        pre_X = X
        show = kwargs.pop("show", True)
        images = []
        for plotter, processor in zip(self.plotters, self.processors):
            fit_trans_kwargs = get_kwargs(kwargs, processor.fit_transform)
            if fit:
                after_X = processor.fit_transform(pre_X, *args, **fit_trans_kwargs)
            else:
                after_X = processor.transform(pre_X, *args, **fit_trans_kwargs)
            if plotter.config._compare:
                path = plotter.plot(pre_X, after_X, *args, show=False, **kwargs)
            else:
                path = plotter.plot(after_X, *args, show=False, **kwargs)
            if isinstance(path, list):
                images.extend(path)
            else:
                images.append(path)
            pre_X = after_X

        if show and is_in_notebook():
            display_image_carousel(images)


class AutoPlotter(_BaseAutoProcessorClass):
    _processor_mapping = PLOTTER_MAPPING

    @classmethod
    def for_dataset(cls, dataset_name, **kwargs):
        """Create a processor for a dataset.

        Args:
            dataset (Bioset): The dataset to create a processor for.

        Returns:
            Processor: The processor for the dataset.
        """

        if is_biosets_available():
            from biosets.packaged_modules import EXPERIMENT_TYPE_ALIAS
        else:
            EXPERIMENT_TYPE_ALIAS = {}

        dataset_name = EXPERIMENT_TYPE_ALIAS.get(dataset_name, dataset_name)
        _plotter_mapping = _LazyAutoMapping(
            DATASET_PLT_TO_MAPPER_NAMES.get(dataset_name), PLOTTER_MAPPING_NAMES
        )
        configs = AutoPlotterConfig.for_dataset(dataset_name)
        procs = []
        for config in configs:
            config_kwargs = get_kwargs(kwargs, config.__class__.__init__)
            procs.append(
                _plotter_mapping[type(config)]._from_config(config, **config_kwargs)
            )

        processors = AutoPreprocessor.for_dataset(dataset_name)

        return PlotterPipeline(procs, processors)

    @classmethod
    def from_processor(
        cls, processor, dataset_name=None, **kwargs
    ) -> Union[PlotterPipeline, BasePlotter]:
        """Create a processor from another processor.

        Args:
            processor (Processor): The processor to create a processor for.

        Returns:
            Processor: The processor for the dataset.
        """

        def get_proc(proc, dataset_name=None, **kwargs):
            dataset_name = dataset_name or proc.config.dataset_name
            if dataset_name:
                if is_biosets_available():
                    from biosets.packaged_modules import EXPERIMENT_TYPE_ALIAS
                else:
                    EXPERIMENT_TYPE_ALIAS = {}
                dataset_name = EXPERIMENT_TYPE_ALIAS.get(dataset_name, dataset_name)
                _plotter_mapping = _LazyAutoMapping(
                    DATASET_PLT_TO_MAPPER_NAMES.get(dataset_name),
                    PLOTTER_MAPPING_NAMES,
                )
                config = AutoPlotterConfig.for_processor(
                    proc.config.processor_name,
                    dataset_name=dataset_name,
                )
            else:
                _plotter_mapping = PLOTTER_MAPPING
                config = AutoPlotterConfig.for_processor(proc.config.processor_name)
            config_kwargs = get_kwargs(kwargs, config.__class__.__init__)
            return _plotter_mapping[type(config)]._from_config(config, **config_kwargs)

        if isinstance(processor, ProcessorPipeline):
            plotters = []
            for proc in processor.steps:
                plotters.append(get_proc(proc[1], dataset_name=dataset_name, **kwargs))
            return PlotterPipeline(plotters, processor.processors)
        elif isinstance(processor, BaseProcessor):
            return get_proc(processor, **kwargs)
        else:
            raise ValueError(
                "processor must be a `biofit.ProcessorPipeline` or `biofit.BaseProcessor`."
            )
