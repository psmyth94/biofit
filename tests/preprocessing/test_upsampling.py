import biofit.config
import pandas as pd
import pytest
from biocore.utils.import_util import is_biosets_available
from biofit.preprocessing.resampling.upsampling import (
    UpSampler,
    UpSamplerConfigForOTU,
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
def test_upsampling(count_data, sample_metadata, format):
    from biocore import DataHandler

    format = format.replace("_cached", "")
    X, y = count_data
    otu_dataset = pd.concat([sample_metadata, X, y], axis=1)
    handler = DataHandler()

    proc = UpSampler(config=UpSamplerConfigForOTU())
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
            data = handler.to_format(otu_dataset, format)
            out = proc.fit_transform(
                data,
                input_columns=list(X.columns),
                target_column=y.columns[0],
                cache_dir=cache_dir,
            )
        labs = handler.to_numpy(out, y.columns[0])
    assert labs.sum() / len(labs) == 0.5
