import pandas as pd
import pytest
from biocore.data_handling import DataHandler
from biocore.utils.import_util import is_biosets_available, is_datasets_available

import biofit.config
from biofit.stat import (
    ColumnMissingnessStat,
    RowMissingnessStat,
)
from biofit.stat.col_missingness.col_missingness import (
    ColumnMissingnessStatConfigForOTU,
)
from biofit.stat.row_missingness.row_missingness import RowMissingnessStatConfigForOTU
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
def test_col_missingness(count_data, sample_metadata, format):
    load_from_cache_file = "_cached" in format
    format = format.replace("_cached", "")

    X, y = count_data
    otu_dataset = pd.concat([sample_metadata, X, y], axis=1)
    proc = ColumnMissingnessStat(
        config=ColumnMissingnessStatConfigForOTU(
            depth=0, load_from_cache_file=load_from_cache_file
        )
    )
    cache_dir = biofit.config.BIOFIT_CACHE_HOME
    input_columns = list(X.columns)
    if format == "numpy":
        data = handler.to_format(X, format)
        out = proc.fit_transform(data, cache_dir=cache_dir)
    else:
        if format == "dataset":
            if not is_datasets_available():
                pytest.skip("test requires datasets")
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
                input_columns=input_columns,
                cache_dir=cache_dir,
            )
    out = handler.to_list(out)
    out = handler.to_list(out)
    out = [out[0] if isinstance(out, list) and len(out) == 1 else out for out in out]

    if isinstance(out[0], list):
        out = out[0]

    expected_values = [9, 13, 10, 6, 14]

    assert out == pytest.approx(expected_values, abs=1e-6)


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
def test_row_missingness(count_data, sample_metadata, format):
    load_from_cache_file = "_cached" in format
    format = format.replace("_cached", "")

    X, y = count_data
    otu_dataset = pd.concat([sample_metadata, X, y], axis=1)
    proc = RowMissingnessStat(
        config=RowMissingnessStatConfigForOTU(
            depth=0, load_from_cache_file=load_from_cache_file
        )
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
    out = [out[0] if isinstance(out, list) and len(out) == 1 else out for out in out]

    if isinstance(out[0], list):
        out = out[0]

    expected_values = [3, 4, 5, 2, 4, 4, 1, 4, 1, 0, 2, 2, 2, 2, 4, 1, 1, 4, 3, 3]
    assert out == pytest.approx(expected_values, abs=1e-6)
