import biofit.config
import pandas as pd
import pytest
from biocore.data_handling import DataHandler
from biocore.utils.import_util import is_biosets_available, is_polars_available
from biofit.preprocessing import LogTransformer

from tests.utils import create_bioset

handler = DataHandler()

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
def test_log_transformer(count_data, sample_metadata, format):
    load_from_cache_file = "_cached" in format
    format = format.replace("_cached", "")
    X, y = count_data
    otu_dataset = pd.concat([sample_metadata, X, y], axis=1)
    proc = LogTransformer(shift=1, load_from_cache_file=load_from_cache_file)
    cache_dir = biofit.config.BIOFIT_CACHE_HOME
    input_columns = list(X.columns)
    if format == "numpy":
        data = handler.to_format(X, format)
        proc.fit_transform(data, cache_dir=cache_dir)
    else:
        if format == "dataset":
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
            proc.fit_transform(otu_dataset, cache_dir=cache_dir)
        else:
            if format == "polars" and not is_polars_available():
                pytest.skip("test requires polars")
            data = handler.to_format(otu_dataset, format)
            proc.fit_transform(
                data,
                input_columns=input_columns,
                cache_dir=cache_dir,
            )
