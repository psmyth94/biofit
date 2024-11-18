import biofit.config
import pandas as pd
import pytest
from biocore.data_handling import DataHandler
from biocore.utils.import_util import is_biosets_available, is_polars_available
from biofit.stat import CorrelationStat
from biofit.stat.correlation.correlation import CorrelationStatConfig

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
def test_correlation(count_data, sample_metadata, format):
    from biocore import DataHandler

    load_from_cache_file = "_cached" in format
    format = format.replace("_cached", "")
    X, y = count_data
    otu_dataset = pd.concat([sample_metadata, X, y], axis=1)
    handler = DataHandler()

    proc = CorrelationStat(
        config=CorrelationStatConfig(load_from_cache_file=load_from_cache_file)
    )
    cache_dir = biofit.config.BIOFIT_CACHE_HOME
    if format == "numpy":
        data = handler.to_format(X, format)
        labs = handler.to_format(y, format)
        out = proc.fit_transform(data, labs, cache_dir=cache_dir)
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
            out = proc.fit_transform(otu_dataset)
        else:
            if format == "polars" and not is_polars_available():
                pytest.skip("test requires polars")
            data = handler.to_format(otu_dataset, format)
            out = proc.fit_transform(
                data,
                input_columns=list(X.columns),
                target_column=y.columns[0],
                cache_dir=cache_dir,
            )
    out = handler.to_list(out)
    if format in ["arrow", "dataset"]:
        out = [out[0] for out in out]

    if isinstance(out[0], list):
        out = out[0]

    expected_vals = [
        0.12309149097933272,
        0.34874291623145787,
        -1.3877787807814457e-17,
        0.09325048082403137,
        0.6081636405595372,
    ]
    assert out == pytest.approx(expected_vals, abs=1e-6)
