import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Type, Union

import numpy as np
import pandas as pd
from biocore.utils.import_util import is_polars_available

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes, sync_backup_config
from biofit.stat import RowMissingnessStat, RowMissingnessStatConfig
from biofit.utils import Unset, logging

from ..filtering import SampleFilter, SampleFilterConfig

if TYPE_CHECKING:
    import polars as pl

logger = logging.get_logger(__name__)


@dataclass
class MinPrevalenceRowSampleFilterConfig(SampleFilterConfig):
    # process description
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
    processor_name: str = field(
        default="min_prevalence_sample_filter", init=False, repr=False
    )

    # default values
    min_prevalence: float = "auto"
    depth: int = None


@dataclass
class MinPrevalenceRowSampleFilterConfigForOTU(MinPrevalenceRowSampleFilterConfig):
    # dataset description
    experiment_name: str = field(default="otu", init=False, repr=False)

    # override default values
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
    depth: int = 0
    min_prevalence: float = "auto"

    def __post_init__(self):
        if self.depth is None:
            self._transform_process_desc = (
                f"Removing samples with <{self.min_prevalence * 100:.0f}% present OTUs"
            )
        elif isinstance(self.min_prevalence, float) and self.min_prevalence < 1:
            self._transform_process_desc = f"Removing samples with <{self.min_prevalence * 100:.0f}% OTUs over {self.depth} counts"
        elif isinstance(self.min_prevalence, (int, float)) and self.min_prevalence >= 1:
            self._transform_process_desc = f"Removing samples with <{self.min_prevalence} OTUs over {self.depth} counts"


@dataclass
class MinPrevalenceRowSampleFilterConfigForSNP(MinPrevalenceRowSampleFilterConfig):
    # dataset description
    experiment_name: str = field(default="snp", init=False, repr=False)

    # override default values
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("GenomicVariant")], init=False, repr=False
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("GenomicVariant")], init=False, repr=False
    )
    depth: int = 0
    min_prevalence: float = 0.012


@dataclass
class MinPrevalenceRowSampleFilterConfigForMaldi(MinPrevalenceRowSampleFilterConfig):
    # dataset description
    experiment_name: str = field(default="maldi", init=False, repr=False)

    # override default values
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("PeakIntensity")], init=False, repr=False
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("PeakIntensity")], init=False, repr=False
    )
    depth: int = 0
    min_prevalence: float = 0.4


class MinPrevalenceSampleFilter(SampleFilter):
    # main config class
    _config_class = MinPrevalenceRowSampleFilterConfig
    config: MinPrevalenceRowSampleFilterConfig

    IQRs = None

    def __init__(
        self,
        config: Optional[MinPrevalenceRowSampleFilterConfig] = None,
        *,
        min_prevalence: Union[str, float] = "auto",
        depth: Union[int, Unset] = None,
        **kwargs,
    ):
        """SampleFilter samples based on the minimum prevalence of features.

        Args:
            config (MinPrevalenceRowSampleFilterConfig, optional):
                The configuration for the filter. Defaults to None. If a configuration is provided, the
                below arguments are ignored. To override the configuration, use the set_params method.
            min_prevalence (Union[str, float], optional):
                The minimum prevalence of features to keep a sample.
                The threshold to which we determine the minimum prevalence percentage or count of present values to maintain a sample or feature.
                Any value passed >= 1 will be the minimum count while any value < 1 will be a percentage of the total values.
                If "auto", the minimum prevalence is calculated as the first quartile minus 1.5 times the interquartile range. Defaults to "auto".
            depth (Union[int, Unset], optional):
                The minimum value that we consider something to be present. Defaults to None.
            **kwargs:
                Additional keyword arguments to pass to the filter. See the ProcessorConfig class for more details.
        """
        super().__init__(
            config=config, min_prevalence=min_prevalence, depth=depth, **kwargs
        )
        row_missingness_config = RowMissingnessStatConfig.from_config(self.config)
        self.missingness = RowMissingnessStat(row_missingness_config)

    @sync_backup_config
    def set_params(self, **kwargs):
        self.config = self.config.replace_defaults(**kwargs)
        row_missingness_config = RowMissingnessStatConfig.from_config(self.config)
        self.missingness = RowMissingnessStat(row_missingness_config)
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
    ) -> "MinPrevalenceSampleFilter":
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
        if self.config.min_prevalence != "auto":
            kwargs["fn_kwargs"]["fn"] = None
        return super()._process_fit_input(input, **kwargs)

    def _fit_polars(self, X: "pl.DataFrame"):
        row_missingness = self.missingness._transform_polars(X)
        row_missingness = row_missingness.get_column(row_missingness.columns[0])
        row_present = X.shape[1] - row_missingness
        iqr = [row_present.quantile(0.25), row_present.quantile(0.75)]
        self.config.min_prevalence = iqr[0] - 1.5 * (iqr[1] - iqr[0])
        return self

    def _partial_fit_polars(self, X: "pl.DataFrame"):
        row_missingness = self.missingness._transform_polars(X)
        row_missingness = row_missingness.get_column(row_missingness.columns[0])
        row_present = X.shape[1] - row_missingness
        iqr = [row_present.quantile(0.25), row_present.quantile(0.75)]
        if self.IQRs is None:
            self.IQRs = [[iqr[0]], [iqr[1]]]
        else:
            self.IQRs[0].append(iqr[0])
            self.IQRs[1].append(iqr[1])
        return self

    def _fit_pandas(self, X: "pd.DataFrame"):
        row_missingness = self.missingness._transform_pandas(X).iloc[:, 0]
        row_present = X.shape[1] - row_missingness
        iqr = [row_present.quantile(0.25), row_present.quantile(0.75)]
        self.config.min_prevalence = iqr[0] - 1.5 * (iqr[1] - iqr[0])
        return self

    def _partial_fit_pandas(self, X: "pd.DataFrame"):
        row_missingness = self.missingness._transform_pandas(X).iloc[:, 0]
        row_present = X.shape[1] - row_missingness
        iqr = [row_present.quantile(0.25), row_present.quantile(0.75)]
        if self.IQRs is None:
            self.IQRs = [[iqr[0]], [iqr[1]]]
        else:
            self.IQRs[0].append(iqr[0])
            self.IQRs[1].append(iqr[1])
        return self

    def _fit_numpy(self, X: np.ndarray):
        row_missingness = self.missingness._transform_numpy(X)[:, 0]
        row_present = X.shape[1] - row_missingness
        iqr = [np.quantile(row_present, 0.25), np.quantile(row_present, 0.75)]
        self.config.min_prevalence = iqr[0] - 1.5 * (iqr[1] - iqr[0])
        return self

    def _partial_fit_numpy(self, X: np.ndarray):
        row_missingness = self.missingness._transform_numpy(X)[:, 0]
        row_present = X.shape[1] - row_missingness
        iqr = [np.quantile(row_present, 0.25), np.quantile(row_present, 0.75)]
        if self.IQRs is None:
            self.IQRs = [[iqr[0]], [iqr[1]]]
        else:
            self.IQRs[0].append(iqr[0])
            self.IQRs[1].append(iqr[1])
        return self

    def _pool_fit_any(self, partial_results: List["MinPrevalenceSampleFilter"]):
        IQRs = [[], []]
        for result in partial_results:
            IQRs[0].extend(result.IQRs[0])
            IQRs[1].extend(result.IQRs[1])
        IQRs[0] = np.mean(IQRs[0])
        IQRs[1] = np.mean(IQRs[1])
        self.config.min_prevalence = IQRs[0] - 1.5 * (IQRs[1] - IQRs[0])
        return self

    def _process_fit_output(self, input, out):
        logger.info(f"Minimum prevalence set to {out.config.min_prevalence * 100:.0f}%")
        return super()._process_fit_output(input, out)

    def _transform_polars(self, X: "pl.DataFrame"):
        if is_polars_available() and "polars" in sys.modules:
            import polars as pl
        total_present: pl.DataFrame = self.missingness._transform_polars(
            X
        ).with_columns((X.shape[1] - pl.col("*")).alias("sum"))

        if self.config.min_prevalence < 1:
            total_present = total_present.with_columns(pl.col("sum") / X.shape[1])
        tests = total_present.with_columns(pl.col("sum") > self.config.min_prevalence)
        if len(tests) == 1:
            return tests[0]
        return tests.to_numpy()[:, 0].tolist()

    def _transform_pandas(self, X: pd.DataFrame):
        total_present = X.shape[1] - self.missingness._transform_pandas(X)
        if self.config.min_prevalence < 1:
            total_present = total_present / X.shape[1]
        tests = total_present > self.config.min_prevalence
        if len(tests) == 1:
            return tests[0]
        return tests.values[:, 0].tolist()

    def _transform_numpy(self, X: np.ndarray):
        total_present = X.shape[1] - self.missingness._transform_numpy(X)
        if self.config.min_prevalence < 1:
            total_present = total_present / X.shape[1]
        tests = total_present > self.config.min_prevalence
        if len(tests) == 1:
            return tests[0]
        return tests[:, 0].tolist()
