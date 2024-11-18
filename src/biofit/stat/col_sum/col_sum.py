"""
This module provides classes for computing the sum of rows and columns in input data.

Classes:
- RowSumStat: Computes the sum of each row in the input data.
- ColumnSumStat: Computes the sum of each column in the input data.
- ColumnStatForOTU: Computes the sum of each column in the OTUAbundance feature.
- RowSumStatForOTU: Computes the sum of each row in the OTUAbundance feature.
- TotalOTUAbundanceStat: Alias for RowSumStatForOTU.

These classes provide methods for transforming and fitting the input data using different libraries such as numpy, pandas, and polars.

Note: The ColumnStatForOTU and RowSumStatForOTU classes are provided for autostat and are equivalent to ColumnSumStat and RowSumStat respectively.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List, Optional, Type

import numpy as np
from biocore import DataHandler

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes
from biofit.utils import logging

from ..stat import Stat, StatConfig

if TYPE_CHECKING:
    import polars as pl

logger = logging.get_logger(__name__)


@dataclass
class ColumnSumStatConfig(StatConfig):
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
    _fit_process_desc: str = field(
        default="Calculating column sums", init=False, repr=False
    )
    _transform_process_desc: str = field(
        default="Calculating column sums", init=False, repr=False
    )
    processor_name: str = field(default="col_sum", init=False, repr=False)


class ColumnSumStat(Stat):
    """
    Computes the sum of each column in the input data.
    """

    # config attributes
    _config_class = ColumnSumStatConfig
    config: ColumnSumStatConfig
    output_dtype = "float64"

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
    ) -> "ColumnSumStat":
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

    def _fit_numpy(self, X: np.ndarray):
        """
        Fits the model to the input data using numpy.

        Args:
            X: Input data as a numpy array.

        Returns:
            Fitted model.
        """
        self.config.sums_ = np.sum(X, axis=0)
        return self

    def _partial_fit_numpy(self, X: np.ndarray):
        """
        Updates the fitted model with additional input data using numpy.

        Args:
            X: Additional input data as a numpy array.

        Returns:
            Updated fitted model.
        """
        if self.config.sums_ is None:
            self.config.sums_ = np.sum(X, axis=0)
        else:
            self.config.sums_ += np.sum(X, axis=0)
        return self

    def _fit_polars(self, X: "pl.DataFrame"):
        """
        Fits the model to the input data using polars.

        Args:
            X: Input data as a polars DataFrame.

        Returns:
            Fitted model.
        """
        self.config.sums_ = X.sum()
        return self

    def _partial_fit_polars(self, X: "pl.DataFrame"):
        """
        Updates the column sums with additional input data using polars.

        Args:
            X: Additional input data as a polars DataFrame.

        Returns:
            Updated fitted model.
        """
        if self.config.sums_ is None:
            self.config.sums_ = X.sum()
        else:
            self.config.sums_ += X.sum()
        return self

    def _pool_fit(self, fitted_processors: List["ColumnSumStat"]):
        """
        Pools the fitted models from different batches.

        Args:
            fitted_processors: List of fitted models.

        Returns:
            Pooled fitted model.
        """
        self.config.sums_ = sum(
            [processor.config.sums_ for processor in fitted_processors]
        )
        return self

    def _process_transform_output(self, output, input, *args, **kwargs):
        return super()._process_transform_output(
            self.config.sums_, input, *args, **kwargs
        )


@dataclass
class ColumnSumStatConfigForOTU(ColumnSumStatConfig):
    # process description
    _transform_process_desc: str = field(
        default="Calculating total OTU abundance for each sample",
        init=False,
        repr=False,
    )
    dataset_name: str = field(default="otu", init=False, repr=False)
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
