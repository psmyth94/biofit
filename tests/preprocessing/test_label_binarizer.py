from unittest.mock import patch

import pandas as pd
import pytest
from biocore.data_handling import DataHandler
from biocore.utils.import_util import is_biosets_available, is_polars_available

import biofit.config
from biofit.preprocessing import LabelBinarizer
from biofit.processing import NonExistentCacheError
from tests.utils import create_bioset

EXPECTED_LABELS = [1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]

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
def test_label_binarizer(count_data_multi_class, sample_metadata, format):
    should_load_from_cache = "_cached" in format
    format = format.replace("_cached", "")

    X, y = count_data_multi_class
    y = pd.DataFrame({y.columns[0]: [["a", "b", "c"][i] for i in y.values.flatten()]})
    otu_dataset_multi_class = pd.concat([sample_metadata, X, y], axis=1)
    cache_dir = biofit.config.BIOFIT_CACHE_HOME
    with patch.object(
        NonExistentCacheError, "__init__", return_value=None
    ) as mock_error:
        if format == "numpy":
            proc = LabelBinarizer(negative_labels="a")
            data = DataHandler.to_format(y, format)
            out = proc.fit_transform(data, cache_dir=cache_dir)
            assert out.shape[1] == 1
            assert out.flatten().tolist() == EXPECTED_LABELS

        else:
            if format == "dataset":
                if not is_biosets_available():
                    pytest.skip("test requires biosets")
                from biosets.features import Abundance, ClassLabel

                otu_dataset_multi_class = create_bioset(
                    X=X,
                    y=y,
                    sample_metadata=sample_metadata,
                    with_feature_metadata=True,
                    feature_type=Abundance,
                    target_type=ClassLabel,
                )

                # column does not need to be specified for dataset format
                proc = LabelBinarizer(negative_labels="a")
                out = proc.fit_transform(otu_dataset_multi_class, cache_dir=cache_dir)
            else:
                if format == "polars" and not is_polars_available():
                    pytest.skip("test requires polars")
                proc = LabelBinarizer(negative_labels="a")
                data = DataHandler.to_format(otu_dataset_multi_class, format)
                out = proc.fit_transform(
                    data, input_columns=y.columns[0], cache_dir=cache_dir
                )
            assert DataHandler.get_shape(out)[1] == 13
            assert (
                DataHandler.to_numpy(out, y.columns[0]).flatten().tolist()
                == EXPECTED_LABELS
            )
        if should_load_from_cache:
            # ensure that NonExistentCacheError was not raised
            mock_error.assert_not_called()
        else:
            mock_error.assert_called_once()
