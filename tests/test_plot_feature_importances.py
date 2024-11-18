import os
import shutil
import unittest

import biofit.config
import numpy as np
import pandas as pd
import pytest
from biofit.visualization.feature_importance import FeatureImportancePlotter

from tests.utils import require_rpy2

pytestmark = pytest.mark.unit


@require_rpy2
class TestFeatureImportancePlotter(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, count_data, sample_metadata, feature_metadata):
        self.X, self.y = count_data
        self.feature_metadata = feature_metadata
        self.sample_metadata = sample_metadata
        self.column_names = self.X.columns
        self.metadata_columns = self.sample_metadata.columns
        self.data = pd.concat([sample_metadata, *count_data], axis=1)

        self.feature_importances = pd.DataFrame(
            {
                "features": self.column_names,
                "importances_1": np.random.randint(0, 255, len(self.column_names)),
                "importances_2": np.random.randint(0, 255, len(self.column_names)),
            }
        )

    def setUp(self):
        self.plotter = FeatureImportancePlotter()

    def tearDown(self):
        # Clean up the cache directory
        cache_dir = biofit.config.BIOFIT_CACHE_HOME
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

    def _assert_plot_output(self, output_path):
        # Retrieve the image paths from output_path
        image_paths = [os.path.join(output_path, f) for f in os.listdir(output_path)]

        # Check that image_paths is not empty
        self.assertIsNotNone(image_paths)
        self.assertTrue(len(image_paths) > 0)

        # Check that the plot files were created and are not empty
        for img_path in image_paths:
            self.assertTrue(os.path.exists(img_path))
            self.assertGreater(os.path.getsize(img_path), 0)

    def test_plot_with_valid_params(self):
        self.plotter.set_params(
            plot_top=10,
            dat_log="log2_1p",
            show_column_names=True,
            scale_legend_title="Abundance",
            column_title="Samples",
            row_title="Features",
            plot_title="Feature Importances",
            feature_meta_name=["features", "my_metadata_str"],
        )

        output_path = biofit.config.BIOFIT_CACHE_HOME

        self.plotter.plot(
            X=self.X,
            y=self.y,
            sample_metadata=self.sample_metadata,
            feature_importances=self.feature_importances,
            feature_metadata=self.feature_metadata,
            path=output_path,
            show=False,
        )

        self._assert_plot_output(output_path)

    def test_plot_minimal_params(self):
        output_path = biofit.config.BIOFIT_CACHE_HOME

        self.plotter.plot(
            self.data,
            feature_importances=self.feature_importances,
            path=output_path,
            show=False,
        )

        self._assert_plot_output(output_path)

    def test_plot_missing_feature_meta_name(self):
        output_path = biofit.config.BIOFIT_CACHE_HOME

        self.plotter.set_params(
            feature_meta_name="non_existent_column",
        )

        with self.assertRaises(ValueError) as context:
            self.plotter.plot(
                X=self.X,
                y=self.y,
                sample_metadata=self.sample_metadata,
                feature_importances=self.feature_importances,
                feature_metadata=self.feature_metadata,
                path=output_path,
                show=False,
            )
        self.assertIn(
            str(context.exception),
            "Feature metadata columns ['non_existent_column'] not found in "
            "feature metadata.",
        )

    def test_plot_with_invalid_dat_log(self):
        self.plotter.set_params(dat_log="invalid_log")

        output_path = biofit.config.BIOFIT_CACHE_HOME

        # Assuming that invalid dat_log values are handled without raising an error
        self.plotter.plot(
            X=self.X,
            y=self.y,
            sample_metadata=self.sample_metadata,
            feature_importances=self.feature_importances,
            feature_metadata=self.feature_metadata,
            path=output_path,
            show=False,
        )

        self._assert_plot_output(output_path)

    def test_plot_with_plot_top_equal_to_num_features(self):
        num_features = len(self.column_names)
        self.plotter.set_params(plot_top=num_features)

        output_path = biofit.config.BIOFIT_CACHE_HOME

        self.plotter.plot(
            X=self.X,
            y=self.y,
            sample_metadata=self.sample_metadata,
            feature_importances=self.feature_importances,
            feature_metadata=self.feature_metadata,
            path=output_path,
            show=False,
        )

        self._assert_plot_output(output_path)

    def test_plot_without_dat_log(self):
        self.plotter.set_params(dat_log=None)

        output_path = biofit.config.BIOFIT_CACHE_HOME

        self.plotter.plot(
            X=self.X,
            y=self.y,
            sample_metadata=self.sample_metadata,
            feature_importances=self.feature_importances,
            feature_metadata=self.feature_metadata,
            path=output_path,
            show=False,
        )

        self._assert_plot_output(output_path)

    def test_plot_with_show_column_names(self):
        self.plotter.set_params(show_column_names=True)

        output_path = biofit.config.BIOFIT_CACHE_HOME

        self.plotter.plot(
            X=self.X,
            y=self.y,
            sample_metadata=self.sample_metadata,
            feature_importances=self.feature_importances,
            feature_metadata=self.feature_metadata,
            path=output_path,
            show=False,
        )

        self._assert_plot_output(output_path)

    def test_plot_with_custom_scale_legend_title(self):
        self.plotter.set_params(scale_legend_title="Custom Legend Title")

        output_path = biofit.config.BIOFIT_CACHE_HOME

        self.plotter.plot(
            X=self.X,
            y=self.y,
            sample_metadata=self.sample_metadata,
            feature_importances=self.feature_importances,
            feature_metadata=self.feature_metadata,
            path=output_path,
            show=False,
        )

        self._assert_plot_output(output_path)

    def test_plot_with_input_columns(self):
        # Select a subset of columns
        selected_columns = self.column_names[:5]
        self.plotter.set_params(input_columns=selected_columns)

        output_path = biofit.config.BIOFIT_CACHE_HOME

        self.plotter.plot(
            X=self.X,
            y=self.y,
            sample_metadata=self.sample_metadata,
            feature_importances=self.feature_importances,
            feature_metadata=self.feature_metadata,
            path=output_path,
            show=False,
        )

        self._assert_plot_output(output_path)

    def test_plot_with_invalid_input_columns(self):
        invalid_columns = ["non_existent_column"]

        output_path = biofit.config.BIOFIT_CACHE_HOME

        with self.assertRaises(ValueError) as context:
            self.plotter.plot(
                X=self.X,
                y=self.y,
                input_columns=invalid_columns,
                sample_metadata=self.sample_metadata,
                feature_importances=self.feature_importances,
                feature_metadata=self.feature_metadata,
                path=output_path,
                show=False,
            )
        self.assertIn(
            "Columns {'non_existent_column'} not found in input dataset",
            str(context.exception),
        )

    def test_plot_with_custom_feature_column(self):
        # Rename the 'features' column in feature_importances
        self.feature_importances.rename(
            columns={"features": "feature_names"}, inplace=True
        )
        self.plotter.set_params(feature_column="feature_names")

        output_path = biofit.config.BIOFIT_CACHE_HOME

        self.plotter.plot(
            X=self.X,
            y=self.y,
            sample_metadata=self.sample_metadata,
            feature_importances=self.feature_importances,
            feature_metadata=self.feature_metadata,
            path=output_path,
            show=False,
        )

        self._assert_plot_output(output_path)

    def test_plot_with_invalid_feature_column(self):
        output_path = biofit.config.BIOFIT_CACHE_HOME

        with self.assertRaises(ValueError) as context:
            self.plotter.plot(
                X=self.X,
                y=self.y,
                feature_column="invalid_column",
                sample_metadata=self.sample_metadata,
                feature_importances=self.feature_importances,
                feature_metadata=self.feature_metadata,
                path=output_path,
                show=False,
            )
        self.assertIn(
            str(context.exception),
            "Feature column 'invalid_column' not found in feature "
            "importances. Please provide the column name found in both "
            "feature importances and feature metadata (if provided).",
        )

    def test_plot_with_no_sample_metadata(self):
        self.plotter.set_params(sample_metadata_columns=None)

        output_path = biofit.config.BIOFIT_CACHE_HOME

        self.plotter.plot(
            X=self.X,
            y=self.y,
            sample_metadata=None,  # sample_metadata is None
            feature_importances=self.feature_importances,
            feature_metadata=self.feature_metadata,
            path=output_path,
            show=False,
        )

        self._assert_plot_output(output_path)

    def test_plot_with_empty_X(self):
        empty_X = pd.DataFrame()

        output_path = biofit.config.BIOFIT_CACHE_HOME

        with self.assertRaises(AssertionError) as context:
            self.plotter.plot(
                X=empty_X,
                y=self.y,
                sample_metadata=self.sample_metadata,
                feature_importances=self.feature_importances,
                feature_metadata=self.feature_metadata,
                path=output_path,
                show=False,
            )
        self.assertIn("Input data has no rows", str(context.exception))

    def test_plot_output_directory(self):
        output_path = biofit.config.BIOFIT_CACHE_HOME

        self.plotter.plot(
            X=self.X,
            y=self.y,
            sample_metadata=self.sample_metadata,
            feature_importances=self.feature_importances,
            feature_metadata=self.feature_metadata,
            path=output_path,
            show=False,
        )

        self._assert_plot_output(output_path)

    def test_plot_show_option(self):
        output_path = biofit.config.BIOFIT_CACHE_HOME

        # Since we cannot check the display in a unit test, we ensure it runs without error
        self.plotter.plot(
            X=self.X,
            y=self.y,
            sample_metadata=self.sample_metadata,
            feature_importances=self.feature_importances,
            feature_metadata=self.feature_metadata,
            path=output_path,
            show=True,
        )

        self._assert_plot_output(output_path)
