import shutil
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from biocore import DataHandler
from biocore.utils.import_util import is_biosets_available, is_polars_available

import biofit.config
from biofit.utils.py_util import set_seed
from biofit.visualization import (
    generate_comparison_histogram,
    plot_correlation,
)
from biofit.visualization.plotting_utils import (
    generate_violin,
    plot_dimension_reduction,
    plot_feature_importance,
    plot_sample_metadata,
)
from tests.utils import create_bioset, require_biosets, require_polars, require_rpy2

SUPPORTED_MODELS = ["lightgbm", "lasso", "random_forest"]  # , "svm"]

FORMATS = [
    "pandas",
    "polars",
    "numpy",
    "arrow",
    "dataset",
]


# Really should be an integration test but we haven't implemented the unit tests
# for each of the plotting classes
pytestmark = pytest.mark.unit


class TestPlottingUtils(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, count_data, sample_metadata):
        self.sample_metadata = sample_metadata
        self.metadata_columns = list(sample_metadata.columns)
        self.X, self.y = count_data
        self.input_columns = list(self.X.columns)
        self.target_column = list(self.y.columns)[0]
        self.data = pd.concat([self.sample_metadata, self.X, self.y], axis=1)

        self.feature_importances = pd.DataFrame(
            {
                "features": self.input_columns,
                "importances_1": np.random.rand(len(self.input_columns)),
                "importances_2": np.random.rand(len(self.input_columns)),
            }
        )
        # Convert the dataset to various formats
        if is_biosets_available():
            from biosets.features import Abundance, BinClassLabel

            self.dataset_all = create_bioset(
                X=self.X,
                y=self.y,
                sample_metadata=self.sample_metadata,
                with_feature_metadata=True,
                feature_type=Abundance,
                target_type=BinClassLabel,
            )

        self.pandas_all = self.data
        if is_polars_available():
            self.polars_all = DataHandler.to_polars(self.data)
        self.arrow_all = DataHandler.to_arrow(self.data)

        # Extract data and target in various formats
        self.numpy_data = DataHandler.to_numpy(self.X)
        self.numpy_target = DataHandler.to_numpy(self.y)

        self.pandas_data = self.X
        self.pandas_target = self.y

        self.polars_data = DataHandler.to_polars(self.X)
        self.polars_target = DataHandler.to_polars(self.y)

        self.arrow_data = DataHandler.to_arrow(self.X)
        self.arrow_target = DataHandler.to_arrow(self.y)

        self.pandas_sample_metadata = self.sample_metadata
        self.polars_sample_metadata = DataHandler.to_polars(self.sample_metadata)
        self.arrow_sample_metadata = DataHandler.to_arrow(self.sample_metadata)

    def setUp(self):
        set_seed(42)
        self.output_dir = Path(biofit.config.BIOFIT_CACHE_HOME)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def tearDown(self):
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    def _assert_plot_outputs(self):
        assert self.output_dir.is_dir()
        assert len([f for f in self.output_dir.iterdir() if f.is_file()]) > 0

    # For dataset format
    @require_rpy2
    def test_feature_importance_dataset(self):
        plot_feature_importance(
            self.feature_importances,
            X=self.data,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_rpy2
    def test_feature_importance_pandas(self):
        plot_feature_importance(
            self.feature_importances,
            X=self.pandas_all,
            input_columns=self.input_columns,
            target_columns=self.target_column,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_rpy2
    def test_feature_importance_pandas_separate(self):
        plot_feature_importance(
            self.feature_importances,
            X=self.pandas_data,
            y=self.pandas_target,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_rpy2
    def test_feature_importance_polars(self):
        plot_feature_importance(
            self.feature_importances,
            X=self.polars_all,
            input_columns=self.input_columns,
            target_columns=self.target_column,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_rpy2
    def test_feature_importance_polars_separate(self):
        plot_feature_importance(
            self.feature_importances,
            X=self.polars_data,
            y=self.polars_target,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_rpy2
    def test_feature_importance_arrow(self):
        plot_feature_importance(
            self.feature_importances,
            X=self.arrow_all,
            input_columns=self.input_columns,
            target_columns=self.target_column,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_rpy2
    def test_feature_importance_arrow_separate(self):
        plot_feature_importance(
            self.feature_importances,
            X=self.arrow_data,
            y=self.arrow_target,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_rpy2
    def test_feature_importance_numpy(self):
        feature_importances = pd.DataFrame(
            {
                "features": [f"col_{i}" for i in range(self.numpy_data.shape[1])],
                "importances_1": np.random.rand(len(self.input_columns)),
                "importances_2": np.random.rand(len(self.input_columns)),
            }
        )

        plot_feature_importance(
            feature_importances,
            X=self.numpy_data,
            y=self.numpy_target,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_rpy2
    def test_sample_metadata_pandas(self):
        input_columns = self.metadata_columns
        plot_sample_metadata(
            self.pandas_all,
            sample_metadata_columns=input_columns,
            outcome_column=self.target_column,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_rpy2
    def test_sample_metadata_polars(self):
        input_columns = self.metadata_columns
        plot_sample_metadata(
            self.polars_all,
            sample_metadata_columns=input_columns,
            outcome_column=self.target_column,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_rpy2
    def test_sample_metadata_arrow(self):
        input_columns = self.metadata_columns
        plot_sample_metadata(
            self.arrow_all,
            sample_metadata_columns=input_columns,
            outcome_column=self.target_column,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_rpy2
    def test_sample_metadata_dataset(self):
        plot_sample_metadata(
            self.data,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    # Test methods for violin plot
    @require_rpy2
    def test_violin_plot_numpy(self):
        generate_violin(
            self.numpy_data,
            self.numpy_target,
            xlab="test",
            ylab="test",
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_rpy2
    def test_violin_plot_pandas(self):
        generate_violin(
            self.pandas_all,
            xlab="test",
            ylab="test",
            column=self.input_columns[0],
            label_name=self.target_column,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_rpy2
    def test_violin_plot_polars(self):
        generate_violin(
            self.polars_all,
            xlab="test",
            ylab="test",
            column=self.input_columns[0],
            label_name=self.target_column,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_rpy2
    def test_violin_plot_arrow(self):
        generate_violin(
            self.arrow_all,
            xlab="test",
            ylab="test",
            column=self.input_columns[0],
            label_name=self.target_column,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_rpy2
    def test_violin_plot_dataset(self):
        generate_violin(
            self.data,
            xlab="test",
            ylab="test",
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    # Test methods for correlation plot
    @require_rpy2
    def test_correlation_plot_numpy(self):
        plot_correlation(
            self.numpy_data,
            self.numpy_target,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_rpy2
    def test_correlation_plot_pandas(self):
        plot_correlation(
            self.pandas_all,
            input_columns=self.input_columns,
            target_column=self.target_column,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_rpy2
    def test_correlation_plot_polars(self):
        plot_correlation(
            self.polars_all,
            input_columns=self.input_columns,
            target_column=self.target_column,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_rpy2
    def test_correlation_plot_arrow(self):
        plot_correlation(
            self.arrow_all,
            input_columns=self.input_columns,
            target_column=self.target_column,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_rpy2
    def test_correlation_plot_dataset(self):
        plot_correlation(
            self.data,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    # Test methods for comparison histogram
    @require_rpy2
    def test_comparison_histogram_numpy(self):
        generate_comparison_histogram(
            self.numpy_data[:, 0],
            self.numpy_data[:, 1],
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_rpy2
    def test_comparison_histogram_pandas(self):
        input_columns = self.input_columns
        generate_comparison_histogram(
            self.pandas_all,
            column1=input_columns[0],
            column2=input_columns[1],
            subplot_title1=input_columns[0],
            subplot_title2=input_columns[1],
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_rpy2
    def test_comparison_histogram_polars(self):
        input_columns = self.input_columns
        generate_comparison_histogram(
            self.polars_all,
            column1=input_columns[0],
            column2=input_columns[1],
            subplot_title1=input_columns[0],
            subplot_title2=input_columns[1],
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_rpy2
    def test_comparison_histogram_arrow(self):
        input_columns = self.input_columns
        generate_comparison_histogram(
            self.arrow_all,
            column1=input_columns[0],
            column2=input_columns[1],
            subplot_title1=input_columns[0],
            subplot_title2=input_columns[1],
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_rpy2
    def test_comparison_histogram_dataset(self):
        input_columns = self.input_columns
        generate_comparison_histogram(
            self.data,
            column1=input_columns[0],
            column2=input_columns[1],
            subplot_title1=input_columns[0],
            subplot_title2=input_columns[1],
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    # Test methods for PCoA plot
    def test_pcoa_plot_numpy(self):
        plot_dimension_reduction(
            self.numpy_data,
            labels=self.numpy_target,
            method="pcoa",
            method_kwargs={"correction": "cailliez"},
            n_components=3,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    def test_pcoa_plot_pandas(self):
        plot_dimension_reduction(
            self.pandas_all,
            method="pcoa",
            method_kwargs={"correction": "cailliez"},
            n_components=3,
            input_columns=self.input_columns,
            label_column=self.target_column,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_polars
    def test_pcoa_plot_polars(self):
        plot_dimension_reduction(
            self.polars_all,
            method="pcoa",
            method_kwargs={"correction": "cailliez"},
            n_components=3,
            input_columns=self.input_columns,
            label_column=self.target_column,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    def test_pcoa_plot_arrow(self):
        plot_dimension_reduction(
            self.arrow_all,
            method="pcoa",
            method_kwargs={"correction": "cailliez"},
            n_components=3,
            input_columns=self.input_columns,
            label_column=self.target_column,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_rpy2
    def test_pcoa_plot_dataset(self):
        plot_dimension_reduction(
            self.dataset_all,
            method="pcoa",
            method_kwargs={"correction": "cailliez"},
            n_components=3,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    # Test methods for PCA plot
    def test_pca_plot_numpy(self):
        plot_dimension_reduction(
            self.numpy_data,
            labels=self.numpy_target,
            method="pca",
            n_components=3,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    def test_pca_plot_pandas(self):
        plot_dimension_reduction(
            self.pandas_all,
            method="pca",
            n_components=3,
            input_columns=self.input_columns,
            label_column=self.target_column,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    def test_pca_plot_polars(self):
        plot_dimension_reduction(
            self.polars_all,
            method="pca",
            n_components=3,
            input_columns=self.input_columns,
            label_column=self.target_column,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    def test_pca_plot_arrow(self):
        plot_dimension_reduction(
            self.arrow_all,
            method="pca",
            n_components=3,
            input_columns=self.input_columns,
            label_column=self.target_column,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    @require_biosets
    def test_pca_plot_dataset(self):
        plot_dimension_reduction(
            self.dataset_all,
            method="pca",
            n_components=3,
            output_dir=self.output_dir.as_posix(),
        )
        self._assert_plot_outputs()

    # # Test methods for feature distribution plot
    # def test_feature_distribution_plot_numpy(self):
    #     plot_feature_distribution(
    #         self.numpy_data,
    #         self.numpy_target,
    #         output_dir=self.output_dir.as_posix(),
    #     )
    #     self._assert_plot_outputs()
    #
    # def test_feature_distribution_plot_pandas(self):
    #     plot_feature_distribution(
    #         self.pandas_all,
    #         output_dir=self.output_dir.as_posix(),
    #     )
    #     self._assert_plot_outputs()
    #
    # def test_feature_distribution_plot_polars(self):
    #     plot_feature_distribution(
    #         self.polars_all,
    #         output_dir=self.output_dir.as_posix(),
    #     )
    #     self._assert_plot_outputs()
    #
    # def test_feature_distribution_plot_arrow(self):
    #     plot_feature_distribution(
    #         self.arrow_all,
    #         output_dir=self.output_dir.as_posix(),
    #     )
    #     self._assert_plot_outputs()
    #
    # def test_feature_distribution_plot_dataset(self):
    #     plot_feature_distribution(
    #         self.otu_dataset,
    #         output_dir=self.output_dir.as_posix(),
    #     )
    #     self._assert_plot_outputs()
    #
    # # Test methods for compare feature distributions plot
    # def test_compare_feature_distribution_plot_numpy(self):
    #     compare_feature_distributions(
    #         self.numpy_data,
    #         self.numpy_data,
    #         output_dir=self.output_dir.as_posix(),
    #     )
    #     self._assert_plot_outputs()
    #
    # def test_compare_feature_distribution_plot_pandas(self):
    #     compare_feature_distributions(
    #         self.pandas_all,
    #         self.pandas_all,
    #         output_dir=self.output_dir.as_posix(),
    #     )
    #     self._assert_plot_outputs()
    #
    # def test_compare_feature_distribution_plot_polars(self):
    #     compare_feature_distributions(
    #         self.polars_all,
    #         self.polars_all,
    #         columns1=self.input_columns,
    #         columns2=self.input_columns,
    #         output_dir=self.output_dir.as_posix(),
    #     )
    #     self._assert_plot_outputs()
    #
    # def test_compare_feature_distribution_plot_arrow(self):
    #     compare_feature_distributions(
    #         self.arrow_all,
    #         self.arrow_all,
    #         output_dir=self.output_dir.as_posix(),
    #     )
    #     self._assert_plot_outputs()
    #
    # def test_compare_feature_distribution_plot_dataset(self):
    #     compare_feature_distributions(
    #         self.otu_dataset,
    #         self.otu_dataset,
    #         output_dir=self.output_dir.as_posix(),
    #     )
    #     self._assert_plot_outputs()
    #
    # # Test methods for barplot
    # def test_barplot_numpy(self):
    #     generate_barplot(
    #         self.numpy_data[:, 0],
    #         self.numpy_target,
    #         output_dir=self.output_dir.as_posix(),
    #     )
    #     self._assert_plot_outputs()
    #
    # def test_barplot_pandas(self):
    #     generate_barplot(
    #         self.pandas_all,
    #         value_name=self.input_columns[0],
    #         label_name=self.target_column,
    #         output_dir=self.output_dir.as_posix(),
    #     )
    #     self._assert_plot_outputs()
    #
    # def test_barplot_polars(self):
    #     generate_barplot(
    #         self.polars_all,
    #         value_name=self.input_columns[0],
    #         label_name=self.target_column,
    #         output_dir=self.output_dir.as_posix(),
    #     )
    #     self._assert_plot_outputs()
    #
    # def test_barplot_arrow(self):
    #     generate_barplot(
    #         self.arrow_all,
    #         value_name=self.input_columns[0],
    #         label_name=self.target_column,
    #         output_dir=self.output_dir.as_posix(),
    #     )
    #     self._assert_plot_outputs()
    #
    # def test_barplot_dataset(self):
    #     generate_barplot(
    #         self.otu_dataset,
    #         value_name=self.input_columns[0],
    #         label_name=self.target_column,
    #         output_dir=self.output_dir.as_posix(),
    #     )
    #     self._assert_plot_outputs()
    #
    # # Test methods for scatterplot
    # def test_scatterplot_numpy(self):
    #     generate_scatterplot(
    #         x=self.numpy_data[:, 0],
    #         y=self.numpy_data[:, 1],
    #         group=self.numpy_target,
    #         output_dir=self.output_dir.as_posix(),
    #     )
    #     self._assert_plot_outputs()
    #
    # def test_scatterplot_pandas(self):
    #     input_columns = self.input_columns
    #     generate_scatterplot(
    #         x=self.pandas_all,
    #         xdata=input_columns[0],
    #         ydata=input_columns[1],
    #         groupby=self.target_column,
    #         output_dir=self.output_dir.as_posix(),
    #     )
    #     self._assert_plot_outputs()
    #
    # def test_scatterplot_polars(self):
    #     input_columns = self.input_columns
    #     generate_scatterplot(
    #         x=self.polars_all,
    #         xdata=input_columns[0],
    #         ydata=input_columns[1],
    #         groupby=self.target_column,
    #         output_dir=self.output_dir.as_posix(),
    #     )
    #     self._assert_plot_outputs()
    #
    # def test_scatterplot_arrow(self):
    #     input_columns = self.input_columns
    #     generate_scatterplot(
    #         x=self.arrow_all,
    #         xdata=input_columns[0],
    #         ydata=input_columns[1],
    #         groupby=self.target_column,
    #         output_dir=self.output_dir.as_posix(),
    #     )
    #     self._assert_plot_outputs()
    #
    # def test_scatterplot_dataset(self):
    #     input_columns = self.input_columns
    #     generate_scatterplot(
    #         x=self.otu_dataset,
    #         xdata=input_columns[0],
    #         ydata=input_columns[1],
    #         groupby=self.target_column,
    #         output_dir=self.output_dir.as_posix(),
    #     )
    #     self._assert_plot_outputs()
