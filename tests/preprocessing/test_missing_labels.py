import biofit.config
import pandas as pd
import pytest
from biocore.utils.import_util import is_biosets_available
from biofit.preprocessing.filtering.missing_labels import MissingLabelsSampleFilter

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
def test_missing_labels(count_data_missing_labels, sample_metadata, format):
    load_from_cache_file = "_cached" in format
    format = format.replace("_cached", "")
    from biocore import DataHandler

    handler = DataHandler()
    X, y = count_data_missing_labels
    otu_dataset_missing_labels = pd.concat([sample_metadata, X, y], axis=1)
    proc = MissingLabelsSampleFilter(
        load_from_cache_file=load_from_cache_file, missing_label="auto"
    )
    cache_dir = biofit.config.BIOFIT_CACHE_HOME
    if format == "numpy":
        data = handler.to_format(y, format)
        out = proc.fit_transform(data, cache_dir=cache_dir)
    else:
        if format == "dataset":
            if not is_biosets_available():
                pytest.skip("test requires biosets")
            from biosets.features import Abundance, BinClassLabel

            otu_dataset_missing_labels = create_bioset(
                X=X,
                y=y,
                sample_metadata=sample_metadata,
                with_feature_metadata=True,
                feature_type=Abundance,
                target_type=BinClassLabel,
            )
            # column does not need to be specified for dataset format
            out = proc.fit_transform(otu_dataset_missing_labels, cache_dir=cache_dir)
        else:
            data = handler.to_format(otu_dataset_missing_labels, format)
            out = proc.fit_transform(
                data, input_columns=y.columns[0], cache_dir=cache_dir
            )
    assert handler.get_shape(out)[0] == 16
