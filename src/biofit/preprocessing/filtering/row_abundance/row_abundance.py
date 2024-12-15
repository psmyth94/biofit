from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Type, Union

import numpy as np
import pandas as pd

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes
from biofit.utils import logging

from ..filtering import SampleFilter, SampleFilterConfig

if TYPE_CHECKING:
    import polars as pl

logger = logging.get_logger(__name__)


@dataclass
class AbundanceSampleFilterConfig(SampleFilterConfig):
    lower_threshold: Union[int, str] = "auto"
    upper_threshold: Union[int, str] = None
    processor_name: str = field(default="row_abundance", init=False, repr=False)
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


@dataclass
class AbundanceSampleFilterConfigForOTU(AbundanceSampleFilterConfig):
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
    experiment_name: str = field(default="otu", init=False, repr=False)
    lower_threshold: Union[int, str] = "auto"
    upper_threshold: Union[int, str] = "auto"

    def __post_init__(self):
        self._transform_process_desc = f"SampleFiltering out samples with less than {self.lower_threshold} total otu abundance"


class AbundanceSampleFilter(SampleFilter):
    _config_class = AbundanceSampleFilterConfig
    config: AbundanceSampleFilterConfig

    IQRs = None

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
    ) -> "AbundanceSampleFilter":
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

    def _process_fit_input(self, input, **kwargs):
        if (
            self.config.lower_threshold != "auto"
            and self.config.upper_threshold != "auto"
        ):
            kwargs["fn_kwargs"]["fn"] = None
        return super()._process_fit_input(input, **kwargs)

    def _fit_polars(self, X: "pl.DataFrame"):
        iqr = [X.sum_horizontal().quantile(0.25), X.sum_horizontal().quantile(0.75)]
        if self.config.lower_threshold == "auto":
            self.config.lower_threshold = iqr[0] - 1.5 * (iqr[1] - iqr[0])
        if self.config.upper_threshold == "auto":
            self.config.upper_threshold = iqr[1] + 1.5 * (iqr[1] - iqr[0])
        return self

    def _partial_fit_polars(self, X: "pl.DataFrame"):
        iqr = [X.sum_horizontal().quantile(0.25), X.sum_horizontal().quantile(0.75)]
        if self.IQRs is None:
            self.IQRs = [[iqr[0]], [iqr[1]]]
        else:
            self.IQRs[0].append(iqr[0])
            self.IQRs[1].append(iqr[1])
        return self

    def _fit_pandas(self, X: "pd.DataFrame"):
        iqr = [X.sum(axis=1).quantile(0.25), X.sum(axis=1).quantile(0.75)]
        if self.config.lower_threshold == "auto":
            self.config.lower_threshold = iqr[0] - 1.5 * (iqr[1] - iqr[0])
        if self.config.upper_threshold == "auto":
            self.config.upper_threshold = iqr[1] + 1.5 * (iqr[1] - iqr[0])
        return self

    def _partial_fit_pandas(self, X: "pd.DataFrame"):
        iqr = [X.sum(axis=1).quantile(0.25), X.sum(axis=1).quantile(0.75)]
        if self.IQRs is None:
            self.IQRs = [[iqr[0]], [iqr[1]]]
        else:
            self.IQRs[0].append(iqr[0])
            self.IQRs[1].append(iqr[1])
        return self

    def _fit_numpy(self, X: np.ndarray):
        iqr = [np.quantile(X.sum(axis=1), 0.25), np.quantile(X.sum(axis=1), 0.75)]
        if self.config.lower_threshold == "auto":
            self.config.lower_threshold = iqr[0] - 1.5 * (iqr[1] - iqr[0])
        if self.config.upper_threshold == "auto":
            self.config.upper_threshold = iqr[1] + 1.5 * (iqr[1] - iqr[0])
        return self

    def _partial_fit_numpy(self, X: np.ndarray):
        iqr = [np.quantile(X.sum(axis=1), 0.25), np.quantile(X.sum(axis=1), 0.75)]
        if self.IQRs is None:
            self.IQRs = [[iqr[0]], [iqr[1]]]
        else:
            self.IQRs[0].append(iqr[0])
            self.IQRs[1].append(iqr[1])
        return self

    def _pool_fit_any(self, partial_results: List["AbundanceSampleFilter"]):
        IQRs = [[], []]
        for result in partial_results:
            IQRs[0].extend(result.IQRs[0])
            IQRs[1].extend(result.IQRs[1])
        IQRs[0] = np.mean(IQRs[0])
        IQRs[1] = np.mean(IQRs[1])
        if self.config.lower_threshold == "auto":
            self.config.lower_threshold = IQRs[0] - 1.5 * (IQRs[1] - IQRs[0])
        if self.config.upper_threshold == "auto":
            self.config.upper_threshold = IQRs[1] + 1.5 * (IQRs[1] - IQRs[0])
        return self

    def _process_fit_output(self, input, out):
        if out.config.lower_threshold and out.config.upper_threshold:
            logger.info(
                f"Using {out.config.lower_threshold} as lower threshold and "
                f"{out.config.upper_threshold} as upper threshold for filtering samples"
            )
        elif out.config.lower_threshold and out.config.upper_threshold:
            logger.info(
                f"Using {out.config.lower_threshold} as lower threshold for filtering samples"
            )
        elif out.config.upper_threshold:
            logger.info(
                f"Using {out.config.upper_threshold} as upper threshold for filtering samples"
            )
        return super()._process_fit_output(input, out)

    def _process_transform_input(self, input, **kwargs):
        if self.config.lower_threshold and self.config.upper_threshold:
            kwargs["desc"] = (
                f"SampleFiltering out examples with less than {self.config.lower_threshold:.2f} and more than {self.config.upper_threshold:.2f} total abundance"
            )
        elif self.config.lower_threshold:
            kwargs["desc"] = (
                f"SampleFiltering out examples with less than {self.config.lower_threshold:.2f} total abundance"
            )
        elif self.config.upper_threshold:
            kwargs["desc"] = (
                f"SampleFiltering out examples with more than {self.config.upper_threshold:.2f} total abundance"
            )

        return super()._process_transform_input(input, **kwargs)

    def _transform_polars(self, X: "pl.DataFrame"):
        row_sum = X.sum_horizontal()
        if self.config.upper_threshold and self.config.lower_threshold:
            return (row_sum > self.config.lower_threshold) & (
                row_sum < self.config.upper_threshold
            )
        elif self.config.upper_threshold:
            return row_sum < self.config.upper_threshold
        elif self.config.lower_threshold:
            return row_sum > self.config.lower_threshold

    def _transform_pandas(self, X: "pd.DataFrame"):
        row_sum = X.sum(axis=1)
        if self.config.upper_threshold and self.config.lower_threshold:
            return (row_sum > self.config.lower_threshold) & (
                row_sum < self.config.upper_threshold
            )
        elif self.config.upper_threshold:
            return row_sum < self.config.upper_threshold
        elif self.config.lower_threshold:
            return row_sum > self.config.lower_threshold

    def _transform_numpy(self, X: np.ndarray):
        row_sum = X.sum(axis=1)
        if self.config.upper_threshold and self.config.lower_threshold:
            return (row_sum > self.config.lower_threshold) & (
                row_sum < self.config.upper_threshold
            )
        elif self.config.upper_threshold:
            return row_sum < self.config.upper_threshold
        elif self.config.lower_threshold:
            return row_sum > self.config.lower_threshold
