import biofit.config
import pandas as pd
import pytest
import sklearn
from biocore.data_handling import DataHandler
from biocore.utils.import_util import is_biosets_available
from biofit.preprocessing import PCAFeatureExtractor
from packaging import version

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
def test_pca_feature_extractor(count_data, sample_metadata, format):
    load_from_cache_file = "_cached" in format
    format = format.replace("_cached", "")
    X, y = count_data
    otu_dataset = pd.concat([sample_metadata, X, y], axis=1)

    proc = PCAFeatureExtractor(load_from_cache_file=load_from_cache_file)
    if version.parse(sklearn.__version__) >= version.parse("1.4.0"):
        expected = [-0.37224401, 1.02218086, -1.25448252, -0.22033803, -0.2220483]
    else:
        # For some reason, the second to last value is positive in previous scikit-learn
        # Probably due to how the sign of the eigenvectors is determined.
        expected = [-0.37224401, 1.02218086, -1.25448252, 0.22033803, -0.2220483]

    input_columns = list(X.columns)
    cache_dir = biofit.config.BIOFIT_CACHE_HOME
    if format == "numpy":
        data = DataHandler.to_format(X, format)
        out = proc.fit_transform(data)
        assert DataHandler.get_shape(out)[1] == 5
        assert DataHandler.to_list(DataHandler.select_row(out, 0)) == pytest.approx(
            expected, abs=1e-3
        )
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
            data = DataHandler.to_format(otu_dataset, format)
            out = proc.fit_transform(
                data, input_columns=input_columns, cache_dir=cache_dir
            )
        # 20 + 8 columns
        assert DataHandler.get_shape(out)[1] == 13
        assert DataHandler.select_row(
            DataHandler.to_numpy(
                DataHandler.select_columns(
                    out, [f"pca_{i}" for i in range(len(input_columns))]
                )
            ),
            0,
        ) == pytest.approx(expected, abs=1e-3)
