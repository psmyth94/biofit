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
from typing import TYPE_CHECKING, List, Type

import numpy as np
import pandas as pd
from biocore.utils.import_util import requires_backends

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes
from biofit.utils import logging

from ..stat import Stat, StatConfig

if TYPE_CHECKING:
    import polars as pl

logger = logging.get_logger(__name__)


@dataclass
class RowSumStatConfig(StatConfig):
    # process description
    _transform_process_desc: str = field(
        default="Calculating row sums", init=False, repr=False
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
    processor_name: str = field(default="row_sum", init=False, repr=False)
    _n_features_out: int = field(default=1, init=False, repr=False)


@dataclass
class RowSumStatConfigForOTU(RowSumStatConfig):
    """Computes the sum of each row in the OTUAbundance feature."""

    # dataset specific attributes
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
    dataset_name: str = field(default="otu", init=False, repr=False)


class RowSumStat(Stat):
    """
    Computes the sum of each row in the input data.
    """

    # config attributes
    _config_class = RowSumStatConfig
    config: RowSumStatConfig
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
    ) -> "RowSumStat":
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

    def _transform_numpy(self, X: np.ndarray):
        """
        Transforms the input data using numpy.

        Args:
            X: Input data as a numpy array.

        Returns:
            Transformed data as a numpy array.
        """
        if len(X.shape) == 1:
            return np.sum(X)[:, None]
        return np.sum(X, axis=1)[:, None]

    def _transform_pandas(self, X: pd.DataFrame):
        """
        Transforms the input data using pandas.

        Args:
            X: Input data as a pandas DataFrame.

        Returns:
            Transformed data as a pandas DataFrame.
        """
        return X.astype("float64").sum(axis=1).to_frame()

    def _transform_polars(self, X: "pl.DataFrame"):
        """
        Transforms the input data using polars.

        Args:
            X: Input data as a polars DataFrame.

        Returns:
            Transformed data as a polars DataFrame.
        """
        requires_backends(self._transform_polars, "polars")
        import polars as pl

        return X.cast(pl.Float64).sum_horizontal().to_frame()
