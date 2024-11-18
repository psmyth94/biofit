import pandas as pd
import pytest
from biocore.data_handling import DataHandler
from biocore.utils.import_util import is_datasets_available

import biofit.config
from biofit.preprocessing.filtering.min_prevalence_sample_filter import (
    MinPrevalenceSampleFilter,
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
def test_max_missing_row(count_data, sample_metadata, format):
    load_from_cache_file = "_cached" in format
    format = format.replace("_cached", "")
    X, y = count_data
    otu_dataset = pd.concat([sample_metadata, X, y], axis=1)

    proc = MinPrevalenceSampleFilter(
        load_from_cache_file=load_from_cache_file, min_prevalence=0.5, depth=0
    )
    cache_dir = biofit.config.BIOFIT_CACHE_HOME
    input_columns = list(X.columns)
    if format == "numpy":
        data = DataHandler.to_format(X, format)
        out = proc.fit_transform(data, cache_dir=cache_dir)
    else:
        if format == "dataset":
            if not is_datasets_available():
                pytest.skip("test requires datasets")
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
        else:
            data = DataHandler.to_format(otu_dataset, format)
            out = proc.fit_transform(
                data,
                input_columns=input_columns,
                cache_dir=cache_dir,
            )
    assert DataHandler.get_shape(out)[0] == 10
