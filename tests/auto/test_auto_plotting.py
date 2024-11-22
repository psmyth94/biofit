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

handler = DataHandler()


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


class MockPlotterConfig(PlotterConfig):
    processor_name = "mock_processor"
    processor_type = "mock_type"

    custom_param = "test_value"


class MockPlotter2Config(PlotterConfig):
    processor_name = "mock_processor2"
    processor_type = "mock_type"

    custom_param = "test_value2"


class MockPlotterConfigForMockExperiment(MockPlotterConfig):
    experiment_name = "mock_dataset"
    custom_param = "test_value3"


class MockPlotter(BasePlotter):
    _config_class = MockPlotterConfig
    processor_name = "mock_processor"

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config)

    def plot(X, *args, **kwargs):
        pass


class MockPlotter2(BasePlotter):
    _config_class = MockPlotter2Config
    processor_name = "mock_processor2"

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config)
        self.custom_param = kwargs.get("custom_param", None)

    def plot(X, *args, **kwargs):
        pass


class MockProcessorConfig(ProcessorConfig):
    processor_name = "mock_processor"
    processor_type = "mock_type"


class MockProcessor2Config(ProcessorConfig):
    processor_name = "mock_processor2"
    processor_type = "mock_type"


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


class TestPlotterPipeline(unittest.TestCase):
    def setUp(self):
        self.plotter_mock = MagicMock(spec=BasePlotter)
        self.processor_mock = MagicMock(spec=BaseProcessor)
        self.plotter_pipeline = PlotterPipeline(
            plotters=[self.plotter_mock], processors=[self.processor_mock]
        )
        self.dataset_mock = MagicMock(_info=MagicMock())

    def test_plot_with_valid_dataset(self):
        # Arrange
        self.processor_mock.fit_transform.return_value = self.dataset_mock
        self.plotter_mock.plot.return_value = "mocked_path"

        # Act
        self.plotter_pipeline.plot(self.dataset_mock, fit=True)

        # Assert
        self.processor_mock.fit_transform.assert_called_once()
        self.plotter_mock.plot.assert_called_once()

    def test_plot_with_invalid_dataset(self):
        with self.assertRaises(ValueError) as context:
            self.plotter_pipeline.plot("invalid_dataset")
        self.assertEqual(
            str(context.exception), "X must be a biofit or huggingface Dataset."
        )

    def test_plot_without_fit(self):
        # Arrange
        self.processor_mock.transform.return_value = self.dataset_mock
        self.plotter_mock.plot.return_value = "mocked_path"

        # Act
        self.plotter_pipeline.plot(self.dataset_mock, fit=False)

        # Assert
        self.processor_mock.transform.assert_called_once()
        self.plotter_mock.plot.assert_called_once()

    def test_plot_with_multiple_processors(self):
        # Arrange
        processor_mock_2 = MagicMock(spec=BaseProcessor)
        processor_mock_2.fit_transform.return_value = self.dataset_mock
        plotter_mock_2 = MagicMock(spec=BasePlotter)
        plotter_mock_2.plot.return_value = "mocked_path_2"
        plotter_pipeline = PlotterPipeline(
            plotters=[self.plotter_mock, plotter_mock_2],
            processors=[self.processor_mock, processor_mock_2],
        )

        # Act
        plotter_pipeline.plot(self.dataset_mock, fit=True)

        # Assert
        self.processor_mock.fit_transform.assert_called_once()
        processor_mock_2.fit_transform.assert_called_once()
        self.plotter_mock.plot.assert_called_once()
        plotter_mock_2.plot.assert_called_once()


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

        AutoPreprocessorConfig.register_pipeline("mock_experiment", ["mock_processor"])
        AutoPreprocessorConfig.register_pipeline(
            "mock_experiment2", ["mock_processor2"]
        )

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
        # Act
        plotter_pipeline = AutoPlotter.for_experiment(dataset_mock)

        # Assert
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
        # Act
        plotter_pipeline = AutoPlotter.for_experiment(dataset_mock)

        # Assert
        self.assertIsInstance(plotter_pipeline, PlotterPipeline)
        self.assertIsInstance(plotter_pipeline.plotters[0], MockPlotter2)
        self.assertIsInstance(plotter_pipeline.plotters[0].config, MockPlotter2Config)
        self.assertEqual(
            plotter_pipeline.plotters[0].config.custom_param, "test_value2"
        )

    def test_loading_with_kwargs_overwriting_default(self):
        # Arrange
        experiment_name = "otu"
        plotter_pipeline = AutoPlotter.for_experiment(
            experiment_name, custom_param="overwritten_value"
        )

        # Assert
        self.assertIsInstance(plotter_pipeline, PlotterPipeline)
        self.assertIsInstance(plotter_pipeline.plotters[0], MockPlotter)
        self.assertEqual(plotter_pipeline.plotters[0].custom_param, "overwritten_value")

    def test_from_processor(self):
        # Arrange
        processor_mock = MagicMock(spec=BaseProcessor)
        processor_mock.config.experiment_name = "test_dataset"
        config_mock = MagicMock(spec=AutoPlotterConfig)
        AutoPlotterConfig.for_processor = MagicMock(return_value=config_mock)
        lazy_auto_mapping_mock = {MockPlotter: MockPlotter(config_mock)}

        # Act
        plotter = AutoPlotter.from_processor(processor_mock)

        # Assert
        self.assertIsInstance(plotter, MockPlotter)
        self.assertIsNotNone(plotter)
