import biofit.config
import pandas as pd
import pytest
from biocore.data_handling import DataHandler
from biocore.utils.import_util import is_biosets_available
from biofit.preprocessing import (
    PCoAFeatureExtractor,
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
def test_pcoa_feature_extractor(count_data, sample_metadata, format):
    load_from_cache_file = "_cached" in format
    format = format.replace("_cached", "")
    X, y = count_data
    otu_dataset = pd.concat([sample_metadata, X, y], axis=1)

    proc = PCoAFeatureExtractor(
        correction="cailliez", load_from_cache_file=load_from_cache_file
    )
    input_columns = list(X.columns)
    cache_dir = biofit.config.BIOFIT_CACHE_HOME
    if format == "numpy":
        data = DataHandler.to_format(X, format)
        out = proc.fit_transform(data, cache_dir=cache_dir)[0]
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
            out = proc.fit_transform(otu_dataset, cache_dir=cache_dir)
            out = DataHandler.to_numpy(
                DataHandler.drop_columns(
                    out, list(sample_metadata.columns) + list(y.columns)
                )
            )[0]
        else:
            data = DataHandler.to_format(otu_dataset, format)
            out = proc.fit_transform(
                data, input_columns=input_columns, cache_dir=cache_dir
            )
            out = DataHandler.to_numpy(
                DataHandler.drop_columns(
                    out, list(sample_metadata.columns) + list(y.columns)
                )
            )[0]

    assert out == pytest.approx(
        [
            1.94211301e-01,
            6.14681757e-01,
            -1.57956243e-01,
            -5.19891966e-01,
            1.95193082e-01,
            1.75318366e-01,
            4.31596285e-02,
            -2.40616503e-01,
            6.36366314e-02,
            -1.08292981e-01,
            1.35341980e-02,
            6.58959194e-16,
            8.61517981e-16,
            3.13058241e-16,
            1.03875001e-02,
            -1.46641936e-01,
            1.47741248e-01,
            7.09769578e-02,
        ],
        abs=1e-2,
    )
