import biofit.config
import pandas as pd
import pytest
from biocore.data_handling import DataHandler
from biocore.utils.import_util import is_biosets_available
from biofit.stat.col_sum.col_sum import ColumnSumStat, ColumnSumStatConfigForOTU
from biofit.stat.row_sum.row_sum import RowSumStat, RowSumStatConfigForOTU

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
def test_col_sum(count_data, sample_metadata, format):
    load_from_cache_file = "_cached" in format
    format = format.replace("_cached", "")
    X, y = count_data
    otu_dataset = pd.concat([sample_metadata, X, y], axis=1)

    proc = ColumnSumStat(
        config=ColumnSumStatConfigForOTU(load_from_cache_file=load_from_cache_file)
    )
    cache_dir = biofit.config.BIOFIT_CACHE_HOME
    if format == "numpy":
        data = DataHandler.to_format(X, format)
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
            data = DataHandler.to_format(otu_dataset, format)
            out = proc.fit_transform(
                data,
                input_columns=list(X.columns),
                cache_dir=cache_dir,
            )
    out = DataHandler.to_list(out)
    out = [out[0] if isinstance(out, list) and len(out) == 1 else out for out in out]

    if isinstance(out[0], list):
        out = out[0]

    expected_values = [16, 12, 14, 15, 9]

    assert out == expected_values


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
def test_row_sum(count_data, sample_metadata, format):
    load_from_cache_file = "_cached" in format
    format = format.replace("_cached", "")
    X, y = count_data
    otu_dataset = pd.concat([sample_metadata, X, y], axis=1)

    proc = RowSumStat(
        config=RowSumStatConfigForOTU(load_from_cache_file=load_from_cache_file)
    )
    cache_dir = biofit.config.BIOFIT_CACHE_HOME
    if format == "numpy":
        data = DataHandler.to_format(X, format)
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
            data = DataHandler.to_format(otu_dataset, format)
            out = proc.fit_transform(
                data,
                input_columns=list(X.columns),
                cache_dir=cache_dir,
            )
    out = DataHandler.to_list(out)
    out = [out[0] if isinstance(out, list) and len(out) == 1 else out for out in out]

    if isinstance(out[0], list):
        out = out[0]

    expected_values = [
        3.0,
        1.0,
        0.0,
        3.0,
        2.0,
        1.0,
        6.0,
        2.0,
        4.0,
        7.0,
        4.0,
        5.0,
        5.0,
        4.0,
        1.0,
        5.0,
        6.0,
        1.0,
        4.0,
        2.0,
    ]
    assert out == pytest.approx(expected_values, abs=1e-6)
