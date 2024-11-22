import biofit.config
import pandas as pd
import pytest
from biocore.data_handling import DataHandler
from biocore.utils.import_util import is_biosets_available, is_polars_available
from biofit.auto import AutoPreprocessor

from tests.utils import create_bioset

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "format",
    [
        "dataset",
        "pandas",
        "polars",
        "numpy",
        "arrow",
        "pandas_cached",
        "polars_cached",
        "numpy_cached",
        "arrow_cached",
        "dataset_cached",
    ],
)
def test_auto_preprocessor(float_data, sample_metadata, format):
    load_from_cache_file = "_cached" in format
    format = format.replace("_cached", "")
    X, y = float_data
    otu_dataset = pd.concat([sample_metadata, X, y], axis=1)
    proc = AutoPreprocessor.for_dataset(
        "snp", load_from_cache_file=load_from_cache_file
    )
    cache_dir = biofit.config.BIOFIT_CACHE_HOME
    input_columns = list(X.columns)
    if format == "numpy":
        data = DataHandler.to_format(X, format)
        proc.fit_transform(data, cache_dir=cache_dir)
    else:
        if format == "dataset":
            if not is_biosets_available():
                pytest.skip("test requires biosets")
            from biosets.features import GenomicVariant, BinClassLabel

            otu_dataset = create_bioset(
                X=X,
                y=y,
                sample_metadata=sample_metadata,
                with_feature_metadata=True,
                feature_type=GenomicVariant,
                target_type=BinClassLabel,
            )

            # column does not need to be specified for dataset format
            proc.fit_transform(otu_dataset, cache_dir=cache_dir)
        else:
            if format == "polars" and not is_polars_available():
                pytest.skip("test requires polars")
            data = DataHandler.to_format(otu_dataset, format)
            proc.fit_transform(
                data,
                input_columns=input_columns,
                cache_dir=cache_dir,
            )
    # TODO: add assertions for the output