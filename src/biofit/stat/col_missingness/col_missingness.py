import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Type, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from biocore import DataHandler

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes, sync_backup_config
from biofit.utils import logging

from ..stat import Stat, StatConfig

if TYPE_CHECKING:
    import polars as pl

logger = logging.get_logger(__name__)


def is_multiprocess_mode():
    return os.getppid() > 1


def _filter_features_pandas(X: pd.DataFrame, depth: Optional[float] = None):
    """
    SampleFilter features in a pandas DataFrame based on their presence in the dataset.

    Args:
        X (pd.DataFrame): The input DataFrame containing the features.
        threshold (float, optional): The minimum required presence of a feature in the dataset. Defaults to 0.5.
        depth (float, optional): The depth threshold for numeric columns. If specified, counts values less than depth. Defaults to None.

    Returns:
        List[int]: A list of column indices that pass the filtering condition.
    """
    total_missing = X.isnull().sum(axis=0)
    if depth is not None:
        total_missing += (X <= depth).sum(axis=0)
    return total_missing.to_frame().transpose()


def _filter_features_numpy(X: np.ndarray, depth: Optional[float] = None):
    total_missing = np.sum(np.isnan(X) | (X is None), axis=0)
    if depth is not None:
        total_missing += np.sum(X <= depth, axis=0)
    return total_missing[np.newaxis, :]


def _filter_features_polars(X: "pl.DataFrame", depth: Optional[float] = None):
    if "polars" not in sys.modules:
        import polars
    else:
        polars = sys.modules["polars"]
    total_missing = X.null_count().sum()
    if depth is not None:
        less_than_depth = X.with_columns(polars.col("*") <= depth).sum()
        total_missing += less_than_depth

    return total_missing


@dataclass
class ColumnMissingnessStatConfig(StatConfig):
    # process description
    transform_process_name: str = field(
        default="Calculating the number of missing values", init=False, repr=False
    )
    processor_name: str = field(default="col_missingness", init=False, repr=False)
    _fit_unused_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES")],
        init=False,
        repr=False,
    )
    _transform_unused_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES")],
        init=False,
        repr=False,
    )

    # config attributes
    depth: Optional[Union[float, int]] = None


@dataclass
class ColumnMissingnessStatConfigForOTU(ColumnMissingnessStatConfig):
    """Computes the sum of each row in the OTUAbundance feature.
    This class is the same as RowSumStatForOTU.
    """

    transform_process_name: str = field(
        default="Calculating species richness", init=False, repr=False
    )
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
    dataset_name: str = field(default="otu", init=False, repr=False)

    depth: Optional[Union[float, int]] = 0


@dataclass
class ColumnMissingnessStatConfigForSNP(ColumnMissingnessStatConfig):
    """Computes the sum of each row in the OTUAbundance feature.
    This class is the same as RowSumStatForOTU.
    """

    transform_process_name: str = field(
        default="Calculating species richness", init=False, repr=False
    )
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
    dataset_name: str = field(default="snp", init=False, repr=False)

    depth: Optional[Union[float, int]] = 0


@dataclass
class ColumnMissingnessStatConfigForMetagenomics(ColumnMissingnessStatConfig):
    """Computes the sum of each row in the OTUAbundance feature.
    This class is the same as RowSumStatForOTU.
    """

    transform_process_name: str = field(
        default="Calculating species richness", init=False, repr=False
    )
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [(get_feature("Abundance"), get_feature("ReadCount"))],
        init=False,
        repr=False,
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [(get_feature("Abundance"), get_feature("ReadCount"))],
        init=False,
        repr=False,
    )
    dataset_name: str = field(default="metagenomics", init=False, repr=False)

    depth: Optional[Union[float, int]] = 0


class ColumnMissingnessStat(Stat):
    # config attributes
    _config_class = ColumnMissingnessStatConfig
    config: ColumnMissingnessStatConfig
    output_dtype = "int64"

    def __init__(
        self,
        depth: Optional[Union[float, int]] = None,
        config: Optional[ColumnMissingnessStatConfig] = None,
        **kwargs,
    ):
        super().__init__(config=config, depth=depth, **kwargs)

    @sync_backup_config
    def set_params(self, **kwargs):
        self.config = self.config.replace_defaults(**kwargs)
        return self

    def fit(
        self,
        X,
        input_columns: SelectedColumnTypes = None,
        raise_if_missing: bool = None,
        cache_output: bool = None,
        load_from_cache_file: bool = None,
        batched: bool = True,
        batch_size: int = 1000,
        batch_format: str = None,
        num_proc: int = None,
        map_kwargs: dict = {"fn_kwargs": {}},
        cache_dir: str = None,
        cache_file_name: str = None,
        fingerprint: str = None,
    ) -> "ColumnMissingnessStat":
        return self._process_fit(
            X,
            raise_if_missing=raise_if_missing,
            cache_output=cache_output,
            load_from_cache_file=load_from_cache_file,
            batched=batched,
            batch_size=batch_size,
            batch_format=batch_format,
            num_proc=num_proc,
            map_kwargs=map_kwargs,
            cache_dir=cache_dir,
            cache_file_name=cache_file_name,
            fingerprint=fingerprint,
        )

    def transform(
        self,
        X,
        input_columns: SelectedColumnTypes = None,
        keep_unused_columns: bool = None,
        raise_if_missing: bool = None,
        cache_output: bool = None,
        cache_dir: str = None,
        cache_file_name: str = None,
        load_from_cache_file: bool = None,
        batched: bool = None,
        batch_size: int = None,
        batch_format: str = None,
        output_format: str = None,
        map_kwargs: dict = None,
        num_proc: int = None,
        fingerprint: str = None,
    ):
        self._input_columns = self._set_input_columns_and_arity(input_columns)
        return self._process_transform(
            X,
            keep_unused_columns=keep_unused_columns,
            raise_if_missing=raise_if_missing,
            cache_output=cache_output,
            cache_dir=cache_dir,
            cache_file_name=cache_file_name,
            load_from_cache_file=load_from_cache_file,
            batched=batched,
            batch_size=batch_size,
            batch_format=batch_format,
            output_format=output_format,
            map_kwargs=map_kwargs,
            num_proc=num_proc,
            fingerprint=fingerprint,
        )

    def fit_transform(
        self,
        X,
        *args,
        input_columns: SelectedColumnTypes = None,
        keep_unused_columns: bool = True,
        raise_if_missing: bool = False,
        cache_output: bool = None,
        load_from_cache_file: bool = None,
        batched: bool = True,
        batch_size: int = 1000,
        batch_format: str = None,
        output_format: str = None,
        map_kwargs: dict = {"fn_kwargs": {}},
        num_proc: int = None,
        cache_dir: str = None,
        cache_file_name: str = None,
        fingerprint: str = None,
    ):
        return self.fit(
            X,
            input_columns=input_columns,
            raise_if_missing=raise_if_missing,
            cache_output=cache_output,
            load_from_cache_file=load_from_cache_file,
            batched=batched,
            batch_size=batch_size,
            batch_format=batch_format,
            num_proc=num_proc,
            map_kwargs=map_kwargs,
            cache_dir=cache_dir,
            cache_file_name=cache_file_name,
            fingerprint=fingerprint,
        ).transform(
            X,
            input_columns=input_columns,
            keep_unused_columns=keep_unused_columns,
            raise_if_missing=raise_if_missing,
            cache_output=cache_output,
            cache_dir=cache_dir,
            cache_file_name=cache_file_name,
            load_from_cache_file=load_from_cache_file,
            batched=batched,
            batch_size=batch_size,
            batch_format=batch_format,
            output_format=output_format,
            map_kwargs=map_kwargs,
            num_proc=num_proc,
            fingerprint=fingerprint,
        )

    def _process_transform_input(self, X, **kwargs):
        kwargs["batch_size"] = DataHandler.get_shape(X)[0]
        return X, kwargs

    def _transform_numpy(self, X: np.ndarray) -> np.ndarray:
        return _filter_features_numpy(X, self.config.depth)

    def _transform_pandas(self, X: pd.DataFrame) -> pd.DataFrame:
        return _filter_features_pandas(X, self.config.depth)

    def _transform_polars(self, X: "pl.DataFrame") -> "pl.DataFrame":
        return _filter_features_polars(X, self.config.depth)

    def _transform_arrow(self, X: pa.Table) -> pa.Table:
        if self.config.depth is not None:
            depth = self.config.depth
            return pa.table(
                {
                    k: [
                        pc.filter(
                            v, pc.or_(pc.less_equal(v, depth), pc.is_null(v))
                        ).length()
                    ]
                    for k, v in zip(X.column_names, X.columns)
                }
            )
        else:
            return pa.table(
                {
                    k: [pc.filter(v, pc.is_null(v)).length()]
                    for k, v in zip(X.column_names, X.columns)
                }
            )
