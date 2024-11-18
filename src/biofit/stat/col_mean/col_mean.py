from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Type

import numpy as np

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes
from biofit.utils import logging

from ..stat import Stat, StatConfig

if TYPE_CHECKING:
    import polars as pl

logger = logging.get_logger(__name__)


@dataclass
class ColumnMeanStatConfig(StatConfig):
    # process description
    _transform_process_desc: str = field(
        default="Calculating column means", init=False, repr=False
    )
    processor_name: str = field(default="col_mean", init=False, repr=False)


@dataclass
class ColumnMeanStatConfigForOTU(ColumnMeanStatConfig):
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
    dataset_name: str = field(default="otu", init=False, repr=False)


class ColumnMeanStat(Stat):
    # config attributes
    _config_class = ColumnMeanStatConfig
    config: ColumnMeanStatConfig

    sums_ = None
    counts_ = 0

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
    ) -> "ColumnMeanStat":
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

    def _fit_numpy(self, X: np.ndarray, y=None):
        self.config.means = np.mean(X, axis=0)
        return self

    def _partial_fit_numpy(self, X: np.ndarray, y=None):
        self.counts_ += X.shape[0]
        if self.sums_ is None:
            self.sums_ = np.sum(X, axis=0)
        else:
            self.sums_ += np.sum(X, axis=0)

        # self.config.means = self.config.sums_ / self.config.counts_
        return self

    def _fit_polars(self, X: "pl.DataFrame"):
        self.config.means = X.mean()
        return self

    def _partial_fit_polars(self, X: "pl.DataFrame"):
        self.counts_ += X.shape[0]
        if self.sums_ is None:
            self.sums_ = X.sum()
        else:
            self.sums_ += X.sum()

        # self.config.means = self.sums_ / self.counts_
        return self

    def _pool_fit(self, fitted_processors: List["ColumnMeanStat"]):
        self.sums_ = sum([p.sums_ for p in fitted_processors])
        self.counts_ = sum([p.counts_ for p in fitted_processors])
        self.config.means = self.sums_ / self.counts_
        return self

    def _process_transform_output(self, output, input, *args, **kwargs):
        return super()._process_transform_output(
            self.config.means, input, *args, **kwargs
        )
