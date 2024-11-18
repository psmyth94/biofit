import biofit.config
import pandas as pd
import pytest
from biocore.data_handling import DataHandler
from biocore.utils.import_util import is_biosets_available, is_rpy2_available
from biofit.preprocessing.feature_selection import (
    MinPrevalenceFeatureSelector,
    MinPrevalenceFeatureSelectorPlotter,
    MinPrevalencePlotterConfigForOTU,
)
from tests.utils import create_bioset

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "format",
    [
        "pandas",
        "polars",
        "numpy",
        "arrow",
        "dataset",
        "pandas_cached",
        "polars_cached",
        "numpy_cached",
        "arrow_cached",
        "dataset_cached",
    ],
)
def test_min_missing(count_data, sample_metadata, format):
    new_format = format.replace("_cached", "")
    X, y = count_data
    otu_dataset = pd.concat([sample_metadata, X, y], axis=1)
    # sparsity is 0.3, 0.5, 0.7 for the three features
    proc = MinPrevalenceFeatureSelector(min_prevalence=0.4, depth=0)
    if is_rpy2_available():
        plotter = MinPrevalenceFeatureSelectorPlotter(
            config=MinPrevalencePlotterConfigForOTU()
        )
    else:
        plotter = None
    cache_dir = biofit.config.BIOFIT_CACHE_HOME
    input_columns = list(X.columns)
    if new_format == "numpy":
        data = DataHandler.to_format(X, new_format)
        out = proc.fit_transform(data, cache_dir=cache_dir)
        if plotter is not None:
            plotter.plot(x1=data, x2=out)
        # has no metadata
        assert DataHandler.get_shape(out)[1] == 3
    else:
        if new_format == "dataset":
            if not is_biosets_available():
                pytest.skip("test requires biosets")
            from biosets.features import Abundance, BinClassLabel

            otu_dataset = create_bioset(
                X=X,
                y=y,
                sample_metadata=sample_metadata,
                with_feature_metadata=True,
                feature_type=Abundance,
                target_type=BinClassLabel,
            )
            # column does not need to be specified for dataset format
            out = proc.fit_transform(otu_dataset, cache_dir=cache_dir)
            if plotter is not None:
                plotter.plot(x1=otu_dataset, x2=out)
        else:
            data = DataHandler.to_format(otu_dataset, new_format)
            out = proc.fit_transform(
                data,
                input_columns=input_columns,
                cache_dir=cache_dir,
            )
            input_columns2 = [
                col for col in DataHandler.get_column_names(out) if col in input_columns
            ]
            if plotter is not None:
                plotter.plot(
                    x1=data,
                    x2=out,
                    input_columns1=input_columns,
                    input_columns2=input_columns2,
                )
        # has metadata
        assert DataHandler.get_shape(out)[1] == 11
    # only calculated when not cached
    if "cached" not in format:
        assert proc.total_missing.tolist() == [9, 13, 10, 6, 14]
