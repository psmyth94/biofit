from dataclasses import dataclass
import unittest
from unittest.mock import MagicMock

import biofit.config
import pandas as pd
import pytest
from biocore.data_handling import DataHandler
from biofit.auto.configuration_auto import AutoPlotterConfig, AutoPreprocessorConfig
from biofit.auto.plotting_auto import AutoPlotter, PlotterPipeline
from biofit.auto.processing_auto import AutoPreprocessor
from biofit.processing import BaseProcessor, ProcessorConfig
from biofit.visualization.plotting import BasePlotter, PlotterConfig

from tests.utils import require_biosets, require_rpy2

pytestmark = pytest.mark.integration


@require_biosets
@require_rpy2
@pytest.mark.parametrize("format", ["dataset", "dataset_cached"])
def test_auto_plotting_otu(count_data, sample_metadata, format):
    from biosets.features import Abundance
    from datasets.features import Features

    format = format.replace("_cached", "")
    X, y = count_data
    otu_dataset = pd.concat([sample_metadata, X, y], axis=1)
    cache_dir = biofit.config.BIOFIT_CACHE_HOME

    otu_dataset = DataHandler.to_bioset(otu_dataset)
    otu_dataset._info.features = Features(
        {
            k: Abundance(dtype="int64") if k in X.columns else v
            for k, v in otu_dataset._info.features.items()
        }
    )
    proc = AutoPlotter.for_experiment("otu", path=cache_dir)
    proc.plot(otu_dataset, path=cache_dir)


@require_biosets
@require_rpy2
@pytest.mark.parametrize("format", ["dataset", "dataset_cached"])
def test_auto_plotting_snp(binary_data, sample_metadata, format):
    from biosets.features import Abundance
    from datasets.features import Features

    format = format.replace("_cached", "")
    X, y = binary_data
    snp_dataset = pd.concat([sample_metadata, X, y], axis=1)
    cache_dir = biofit.config.BIOFIT_CACHE_HOME

    snp_dataset = DataHandler.to_bioset(snp_dataset)
    snp_dataset._info.features = Features(
        {
            k: Abundance(dtype="int64") if k in X.columns else v
            for k, v in snp_dataset._info.features.items()
        }
    )
    proc = AutoPlotter.for_experiment("snp", path=cache_dir)
    proc.plot(snp_dataset, path=cache_dir)


@dataclass
class MockPlotterConfig(PlotterConfig):
    processor_name: str = "mock_processor"
    processor_type: str = "mock_type"

    custom_param: str = "test_value"


@dataclass
class MockPlotter2Config(PlotterConfig):
    processor_name: str = "mock_processor2"
    processor_type: str = "mock_type"

    custom_param: str = "test_value2"


@dataclass
class MockPlotterConfigForMockExperiment(MockPlotterConfig):
    experiment_name: str = "mock_dataset"
    custom_param: str = "test_value3"


@dataclass
class MockPlotter(BasePlotter):
    _config_class = MockPlotterConfig
    processor_name = "mock_processor"

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)

    def plot(X, *args, **kwargs):
        pass


@dataclass
class MockPlotter2(BasePlotter):
    _config_class = MockPlotter2Config
    processor_name = "mock_processor2"

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config)
        self.custom_param = kwargs.get("custom_param", None)

    def plot(X, *args, **kwargs):
        pass


@dataclass
class MockProcessorConfig(ProcessorConfig):
    processor_name: str = "mock_processor"
    processor_type: str = "mock_type"


@dataclass
class MockProcessor2Config(ProcessorConfig):
    processor_name: str = "mock_processor2"
    processor_type: str = "mock_type"


class MockProcessor(BaseProcessor):
    _config_class = MockProcessorConfig

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config)
        self.custom_param = kwargs.get("custom_param", None)

    def fit_transform(X, *args, **kwargs):
        pass


class MockProcessor2(BaseProcessor):
    _config_class = MockProcessor2Config

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config)
        self.custom_param = kwargs.get("custom_param", None)

    def fit_transform(X, *args, **kwargs):
        pass


class TestAutoPlotter(unittest.TestCase):
    def setUp(self):
        ## Processor name -> Config Class
        AutoPreprocessorConfig.register_experiment(
            "mock_experiment",
            ["mock_processor", "mock_processor2"],
            [MockProcessorConfig, MockProcessor2Config],
        )

        AutoPlotterConfig.register_experiment(
            "mock_experiment",
            ["mock_processor", "mock_processor2"],
            [MockPlotterConfigForMockExperiment, MockPlotter2Config],
        )

        AutoPreprocessorConfig.register_experiment(
            "mock_experiment2",
            ["mock_processor", "mock_processor2"],
            [MockProcessorConfig, MockProcessor2Config],
        )

        AutoPlotterConfig.register_experiment(
            "mock_experiment2",
            ["mock_processor", "mock_processor2"],
            [MockPlotterConfig, MockPlotter2Config],
        )

        ## Config Class -> Processor/Plotter Class
        AutoPlotter.register_experiment(
            "mock_experiment",
            [
                MockPlotterConfigForMockExperiment,
                MockPlotter2Config,
            ],
            [MockPlotter, MockPlotter2],
        )
        AutoPreprocessor.register_experiment(
            "mock_experiment",
            [MockProcessor, MockProcessor2],
            [MockProcessorConfig, MockProcessor2Config],
        )
        AutoPlotter.register_experiment(
            "mock_experiment2",
            [
                MockPlotterConfigForMockExperiment,
                MockPlotter2Config,
            ],
            [MockPlotter, MockPlotter2],
        )
        AutoPreprocessor.register_experiment(
            "mock_experiment2",
            [MockProcessor, MockProcessor2],
            [MockProcessorConfig, MockProcessor2Config],
        )

        AutoPlotter.register(
            MockPlotterConfig, MockPlotter
        )

        AutoPreprocessorConfig.register_pipeline("mock_experiment", ["mock_processor"])
        AutoPreprocessorConfig.register_pipeline(
            "mock_experiment2", ["mock_processor2"]
        )

    def tearDown(self):
        AutoPreprocessor.unregister_experiment("mock_experiment")
        AutoPlotter.unregister_experiment("mock_experiment")
        AutoPreprocessorConfig.unregister_experiment("mock_experiment")
        AutoPlotterConfig.unregister_experiment("mock_experiment")
        AutoPreprocessor.unregister_experiment("mock_experiment2")
        AutoPlotter.unregister_experiment("mock_experiment2")
        AutoPreprocessorConfig.unregister_experiment("mock_experiment2")
        AutoPlotterConfig.unregister_experiment("mock_experiment2")

        AutoPreprocessorConfig.unregister_pipeline("mock_experiment")
        AutoPreprocessorConfig.unregister_pipeline("mock_experiment2")

    def test_key_already_exist(self):
        with self.assertRaises(ValueError) as context:
            AutoPreprocessorConfig.register_experiment(
                "mock_experiment",
                ["mock_processor3", "mock_processor3"],
                [object, object],
            )
        self.assertEqual(
            str(context.exception),
            "'mock_processor3' is already used by a processor config, pick another name.",
        )

    @require_biosets
    def test_for_dataset(self):
        from biosets import Bioset

        experiment_name = "mock_experiment"
        dataset_mock = MagicMock(spec=Bioset, _info=MagicMock())
        dataset_mock._info.builder_name = experiment_name
        plotter_pipeline = AutoPlotter.for_experiment(dataset_mock)

        self.assertIsInstance(plotter_pipeline, PlotterPipeline)
        self.assertIsInstance(plotter_pipeline.plotters[0], MockPlotter)
        self.assertIsInstance(
            plotter_pipeline.plotters[0].config, MockPlotterConfigForMockExperiment
        )
        self.assertEqual(
            plotter_pipeline.plotters[0].config.custom_param, "test_value3"
        )

        experiment_name = "mock_experiment2"
        dataset_mock = MagicMock(spec=Bioset, _info=MagicMock())
        dataset_mock._info.builder_name = experiment_name
        plotter_pipeline = AutoPlotter.for_experiment(dataset_mock)

        self.assertIsInstance(plotter_pipeline, PlotterPipeline)
        self.assertIsInstance(plotter_pipeline.plotters[0], MockPlotter2)
        self.assertIsInstance(plotter_pipeline.plotters[0].config, MockPlotter2Config)
        self.assertEqual(
            plotter_pipeline.plotters[0].config.custom_param, "test_value2"
        )

    def test_loading_with_kwargs_overwriting_default(self):
        experiment_name = "mock_experiment"
        plotter_pipeline = AutoPlotter.for_experiment(
            experiment_name, custom_param="overwritten_value"
        )

        self.assertIsInstance(plotter_pipeline, PlotterPipeline)
        self.assertIsInstance(plotter_pipeline.plotters[0], MockPlotter)
        self.assertEqual(
            plotter_pipeline.plotters[0].config.custom_param, "overwritten_value"
        )

    def test_from_processor(self):
        plotter = AutoPlotter.from_processor(MockProcessor())

        self.assertIsInstance(plotter, MockPlotter)
        self.assertIsNotNone(plotter)
