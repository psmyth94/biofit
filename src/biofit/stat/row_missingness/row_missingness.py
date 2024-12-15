import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Type

import numpy as np
import pandas as pd

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes
from biofit.utils import logging

from ..stat import Stat, StatConfig

if TYPE_CHECKING:
    import polars as pl

logger = logging.get_logger(__name__)


def is_multiprocess_mode():
    return os.getppid() > 1


def _filter_samples_pandas(X: pd.DataFrame, depth: Optional[float] = None):
    """
    SampleFilter samples in a pandas DataFrame based on their presence in the dataset.

    Args:
        X (pd.DataFrame): The input DataFrame containing the samples.
        min_sample_presence (float, optional): The minimum required presence of a sample in the dataset. Defaults to 0.5.
        depth (float, optional): The depth threshold for numeric columns. If specified, counts values less than depth. Defaults to None.

    Returns:
        List[int]: A list of row indices that pass the filtering condition.
    """
    total_missing = X.isnull().sum(axis=1)
    if depth is not None:
        numeric_rows = X.select_dtypes(include=np.number).index
        total_missing[numeric_rows] += (X.loc[numeric_rows] <= depth).sum(axis=1)
    return total_missing.to_frame()


def _filter_samples_numpy(X: np.ndarray, depth: Optional[float] = None):
    total_missing = np.sum(np.isnan(X) | (X is None), axis=1)
    if depth is not None:
        total_missing += np.sum(X <= depth, axis=1)
    return total_missing[:, np.newaxis]


def _filter_samples_polars(X: "pl.DataFrame", depth: Optional[float] = None):
    if "polars" not in sys.modules:
        import polars
    else:
        polars = sys.modules["polars"]
    total_missing = X.with_columns(
        polars.col("*").is_null() | polars.col("*").is_nan()
    ).sum_horizontal()
    if depth is not None:
        total_missing += X.with_columns(polars.col("*") <= depth).sum_horizontal()
    return total_missing.to_frame()


@dataclass
class RowMissingnessStatConfig(StatConfig):
    # process description
    _transform_process_desc: str = field(
        default="Calculating the number of missing values for each sample",
        init=False,
        repr=False,
    )
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
    processor_name: str = field(default="row_missingness", init=False, repr=False)
    _n_features_out: int = field(default=1, init=False, repr=False)

    # config attributes
    depth: Optional[float] = None


@dataclass
class RowMissingnessStatConfigForOTU(RowMissingnessStatConfig):
    """Computes the sum of each row in the OTUAbundance feature.
    This class is the same as TotalOTUAbundanceStat.
    It is provided for autostat
    """

    # dataset specific attributes
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
    transform_process_name: str = field(
        default="Calculating sample richness", init=False, repr=False
    )
    experiment_name: str = field(default="otu", init=False, repr=False)

    # config attributes
    depth = 100


@dataclass
class RowMissingnessStatConfigForSNP(RowMissingnessStatConfig):
    """Computes the sum of each row in the OTUAbundance feature.
    This class is the same as TotalOTUAbundanceStat.
    It is provided for autostat
    """

    # dataset specific attributes
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("GenomicVariant")], init=False, repr=False
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("GenomicVariant")], init=False, repr=False
    )
    transform_process_name: str = field(
        default="Calculating sample richness", init=False, repr=False
    )
    experiment_name: str = field(default="snp", init=False, repr=False)

    # config attributes
    depth = 100


@dataclass
class RowMissingnessStatConfigForMetagenomics(RowMissingnessStatConfig):
    """Computes the sum of each row in the OTUAbundance feature.
    This class is the same as TotalOTUAbundanceStat.
    It is provided for autostat
    """

    # dataset specific attributes
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
    transform_process_name: str = field(
        default="Calculating sample richness", init=False, repr=False
    )

    # config attributes
    depth: int = 100


class RowMissingnessStat(Stat):
    # feature attributes
    one_to_one_features = False
    n_features_out = 1

    # config attributes
    _config_class = RowMissingnessStatConfig
    config: RowMissingnessStatConfig

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
    ) -> "RowMissingnessStat":
        self.config._input_columns = self._set_input_columns_and_arity(input_columns)
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

    def _transform_numpy(self, X: np.ndarray) -> np.ndarray:
        return _filter_samples_numpy(X, self.config.depth)

    def _transform_pandas(self, X: pd.DataFrame) -> pd.DataFrame:
        return _filter_samples_pandas(X, self.config.depth)

    def _transform_polars(self, X: "pl.DataFrame") -> "pl.DataFrame":
        return _filter_samples_polars(X, self.config.depth)
