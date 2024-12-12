import unittest
from unittest.mock import MagicMock

import pytest
from biofit.auto.plotting_auto import PlotterPipeline
from biofit.processing import BaseProcessor, ProcessorConfig
from biofit.visualization.plotting import BasePlotter, PlotterConfig

from tests.utils import require_biosets

pytestmark = pytest.mark.integration


@require_biosets
class TestPlotterPipeline(unittest.TestCase):
    def setUp(self):
        from biosets import Bioset

        self.plotter_mock = MagicMock(
            spec=BasePlotter,
            config=MagicMock(
                processor_name="mock_processor",
                processor_type="mock_type",
                custom_param="test_value",
                spec=PlotterConfig,
            ),
        )
        self.processor_mock = MagicMock(
            spec=BaseProcessor,
            config=MagicMock(
                processor_name="mock_processor",
                processor_type="mock_type",
                spec=ProcessorConfig,
            ),
        )
        self.plotter_mock_2 = MagicMock(
            spec=BasePlotter,
            config=MagicMock(
                processor_name="mock_processor2",
                processor_type="mock_type",
                custom_param="test_value2",
                spec=PlotterConfig,
            ),
        )
        self.processor_mock_2 = MagicMock(
            spec=BaseProcessor,
            config=MagicMock(
                processor_name="mock_processor2",
                processor_type="mock_type",
                spec=ProcessorConfig,
            ),
        )
        self.plotter_pipeline = PlotterPipeline(
            plotters=[self.plotter_mock], processors=[self.processor_mock]
        )
        self.dataset_mock = MagicMock(
            _info=MagicMock(builder_name="mock_experiment"),
            spec=Bioset,
        )

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
            str(context.exception), "X must be a Bioset or huggingface Dataset."
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
        self.processor_mock_2.fit_transform.return_value = self.dataset_mock
        self.plotter_mock_2.plot.return_value = "mocked_path_2"
        plotter_pipeline = PlotterPipeline(
            plotters=[self.plotter_mock, self.plotter_mock_2],
            processors=[self.processor_mock, self.processor_mock_2],
        )

        # Act
        plotter_pipeline.plot(self.dataset_mock, fit=True)

        # Assert
        self.processor_mock.fit_transform.assert_called_once()
        self.processor_mock_2.fit_transform.assert_called_once()
        self.plotter_mock.plot.assert_called_once()
        self.plotter_mock_2.plot.assert_called_once()
