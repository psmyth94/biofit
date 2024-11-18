import biofit.config
import pandas as pd
import pytest
from biocore.data_handling import DataHandler
from biocore.utils.import_util import is_biosets_available
from biofit.stat.distance import DistanceStat
from biofit.stat.distance.distance import DistanceStatConfigForOTU

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
def test_distance(count_data, sample_metadata, format):
    from biocore import DataHandler

    load_from_cache_file = "_cached" in format
    format = format.replace("_cached", "")
    X, y = count_data
    otu_dataset = pd.concat([sample_metadata, X, y], axis=1)
    handler = DataHandler()

    proc = DistanceStat(
        config=DistanceStatConfigForOTU(load_from_cache_file=load_from_cache_file)
    )
    cache_dir = biofit.config.BIOFIT_CACHE_HOME
    if format == "numpy":
        data = handler.to_format(X, format)
        out = proc.fit_transform(data, cache_dir=cache_dir)
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
            data = handler.to_format(otu_dataset, format)
            out = proc.fit_transform(
                data,
                input_columns=list(X.columns),
                cache_dir=cache_dir,
            )
    out = handler.to_list(out)
    if format in ["arrow", "dataset"]:
        out = [out[0] for out in out]

    if isinstance(out[0], list):
        out = out[0]

    expected_vals = [
        0.0,
        1.0,
        1.0,
        0.3333333333333333,
        1.0,
        0.5,
        0.7777777777777778,
        0.6,
        0.42857142857142855,
        0.4,
        0.7142857142857143,
        0.75,
        0.25,
        0.42857142857142855,
        1.0,
        0.75,
        0.7777777777777778,
        1.0,
        0.7142857142857143,
        0.6,
    ]
    assert out == pytest.approx(expected_vals, abs=1e-6)
