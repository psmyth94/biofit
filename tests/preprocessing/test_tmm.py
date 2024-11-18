import biofit.config
import pandas as pd
import pytest
from biocore.data_handling import DataHandler
from biocore.utils.import_util import is_biosets_available, is_polars_available
from biofit.preprocessing import TMMScaler

from tests.utils import create_bioset, require_rpy2

FORMATS = [
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
]

pytestmark = pytest.mark.unit


@require_rpy2
@pytest.mark.parametrize("format", FORMATS)
def test_otu_tmm(count_data, sample_metadata, format):
    format = format.replace("_cached", "")
    X, y = count_data
    otu_dataset = pd.concat([sample_metadata, X, y], axis=1)
    proc = TMMScaler(install_missing=True)
    cache_dir = biofit.config.BIOFIT_CACHE_HOME
    input_columns = list(X.columns)
    if format == "numpy":
        data = DataHandler.to_format(X, format)
        trans_data = proc.fit_transform(data, cache_dir=cache_dir)
        trans_data = pd.DataFrame(trans_data, columns=input_columns)
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
            trans_data = proc.fit_transform(otu_dataset, cache_dir=cache_dir)
        else:
            if format == "polars" and not is_polars_available():
                pytest.skip("test requires polars")
            data = DataHandler.to_format(otu_dataset, format)
            trans_data = proc.fit_transform(
                data,
                input_columns=input_columns,
                cache_dir=cache_dir,
            )
        trans_data = DataHandler.to_pandas(trans_data)[input_columns]

    expected_means = [
        16.29848284590347,
        16.109477689114623,
        16.1056032673168,
        16.26133423278604,
        16.365374835244474,
        16.52574914648284,
        16.03496933879711,
        16.333477800711826,
        16.329265689665576,
        16.414420051652822,
        16.327363474257353,
        16.3420904314305,
        16.51850994922633,
        16.372262272154607,
        16.643263063183195,
        16.429082863879305,
        16.62725573408411,
        16.268096219909324,
        16.187416726134824,
        16.238862083800324,
    ]

    assert trans_data.mean().tolist() == pytest.approx(expected_means, rel=1e-6)
