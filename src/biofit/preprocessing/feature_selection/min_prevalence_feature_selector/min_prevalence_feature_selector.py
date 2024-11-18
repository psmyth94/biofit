"""
Feature selector that filters features based on the number of samples they are present in.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Type

import numpy as np
from biocore import DataHandler

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes, sync_backup_config
from biofit.stat import ColumnMissingnessStat, ColumnMissingnessStatConfig
from biofit.utils import logging

from ..feature_selection import FeatureSelector, FeatureSelectorConfig

logger = logging.get_logger(__name__)

FILTER_FEATURES_DOCSTRING = """
SampleFilter features based on the number of samples they are present in.

Args:
    X: Input data
    min_prevalence: Minimum number of samples a feature must be present in to be kept.

Returns:
    SampleFiltered data.
"""


def _filter_features(nrows: int, total_missing, min_prevalence, cols=None):
    """
    SampleFilter features in a pandas DataFrame based on their presence in the dataset.

    Args:
        X (pd.DataFrame): The input DataFrame containing the features.
        min_prevalence (float, optional): The minimum required presence of a feature in the dataset.
        depth (float, optional): The minimum value to be considered as present.

    Returns:
        List[int]: A list of column indices that pass the filtering condition.
    """
    # keep features that are present in at least min_prevalence of the rows
    col_to_keep = total_missing <= (nrows * (1 - min_prevalence))
    if cols and len(cols) == len(col_to_keep):
        return [c for c, b in zip(cols, col_to_keep) if b]
    else:
        return [i for i, b in enumerate(col_to_keep) if b]


@dataclass
class MinPrevalenceFeatureSelectorConfig(FeatureSelectorConfig):
    processor_name: str = field(
        default="min_prevalence_feature_selector", init=False, repr=False
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

    min_prevalence: float = 0.5
    depth: int = None

    def __post_init__(self):
        if self.depth is None:
            if isinstance(self.min_prevalence, float) and self.min_prevalence < 1:
                self._fit_process_desc = (
                    "Removing features that are present in less than "
                    f"{self.min_prevalence * 100:.0f}% of samples"
                )
            elif (
                isinstance(self.min_prevalence, (int, float))
                and self.min_prevalence >= 1
            ):
                self._fit_process_desc = f"Removing features that are present in less than {self.min_prevalence} samples"

        elif isinstance(self.min_prevalence, float) and self.min_prevalence < 1:
            self._fit_process_desc = (
                f"Removing features with <{self.min_prevalence * 100:.0f}% "
                f"samples above {self.depth} counts"
            )
        elif isinstance(self.min_prevalence, (int, float)) and self.min_prevalence >= 1:
            self._fit_process_desc = (
                f"Removing features with <{self.min_prevalence} "
                f"samples above {self.depth} counts"
            )


@dataclass
class MinPrevalenceFeatureSelectorConfigForMetagenomics(
    MinPrevalenceFeatureSelectorConfig
):
    dataset_name: str = field(default="metagenomics", init=False, repr=False)
    _fit_input_feature_types: List[Tuple[Type, Type]] = field(
        default_factory=lambda: [(get_feature("Abundance"), get_feature("ReadCount"))],
        init=False,
        repr=False,
    )
    _transform_input_feature_types: List[Tuple[Type, Type]] = field(
        default_factory=lambda: [(get_feature("Abundance"), get_feature("ReadCount"))],
        init=False,
        repr=False,
    )
    depth: int = 100
    min_prevalence: float = 0.01


@dataclass
class MinPrevalenceFeatureSelectorConfigForOTU(MinPrevalenceFeatureSelectorConfig):
    dataset_name: str = field(default="otu", init=False, repr=False)
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
    depth: int = 0
    min_prevalence: float = 0.01

    def __post_init__(self):
        if self.depth is None:
            if isinstance(self.min_prevalence, float) and self.min_prevalence < 1:
                self._fit_process_desc = (
                    "Removing OTUs that are present in less than "
                    f"{self.min_prevalence * 100:.0f}% of samples"
                )
            elif (
                isinstance(self.min_prevalence, (int, float))
                and self.min_prevalence >= 1
            ):
                self._fit_process_desc = f"Removing OTUs that are present in less than {self.min_prevalence} samples"

        elif isinstance(self.min_prevalence, float) and self.min_prevalence < 1:
            self._fit_process_desc = (
                f"Removing OTUs with <{self.min_prevalence * 100:.0f}% "
                f"samples above {self.depth} counts"
            )
        elif isinstance(self.min_prevalence, (int, float)) and self.min_prevalence >= 1:
            self._fit_process_desc = (
                f"Removing OTUs with <{self.min_prevalence} "
                f"samples above {self.depth} counts"
            )


@dataclass
class MinPrevalenceFeatureSelectorConfigForSNP(MinPrevalenceFeatureSelectorConfig):
    dataset_name: str = field(default="snp", init=False, repr=False)
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("GenomicVariant")], init=False, repr=False
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("GenomicVariant")], init=False, repr=False
    )
    depth: int = 0
    min_prevalence: float = 0.012


class MinPrevalenceFeatureSelector(FeatureSelector):
    """
    Feature selector that filters features based on the number of samples they are present in.

    - config:
    - min_prevalence (Union[str, float], optional):
        The minimum prevalence of features to keep a sample.
        The threshold to which we determine the minimum prevalence percentage or count of present values to maintain a sample or feature.
        Any value passed >= 1 will be the minimum count while any value < 1 will be a percentage of the total values.
        If "auto", the minimum prevalence is calculated as the first quartile minus 1.5 times the interquartile range. Defaults to "auto".
    - depth (Union[int, Unset], optional):
        The minimum value that we consider something to be present. Defaults to None.
    """

    config_class = MinPrevalenceFeatureSelectorConfig
    config: MinPrevalenceFeatureSelectorConfig

    def __init__(
        self,
        min_prevalence: float = 0.5,
        depth: int = None,
        config: Optional[MinPrevalenceFeatureSelectorConfig] = None,
        **kwargs,
    ):
        super().__init__(
            config=config, depth=depth, min_prevalence=min_prevalence, **kwargs
        )
        self = self.set_params()

    @sync_backup_config
    def set_params(self, **kwargs):
        if len(kwargs) > 0:
            self.config = self.config.replace_defaults(**kwargs)
        col_missingness_config = ColumnMissingnessStatConfig.from_config(self.config)
        col_missingness_config = col_missingness_config.replace_defaults(**kwargs)
        self.missingness = ColumnMissingnessStat(config=col_missingness_config)
        self.total_missing = None
        self._input_columns = None
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
    ) -> "MinPrevalenceFeatureSelector":
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
        self.height = DataHandler.get_shape(input)[0]
        return input, kwargs

    def _fit_numpy(self, X):
        self.total_missing = self.missingness._transform_numpy(X).flatten()
        return self

    def _partial_fit_numpy(self, X):
        if self.total_missing is None:
            self.total_missing = self.missingness._transform_numpy(X).flatten()
        else:
            self.total_missing += self.missingness._transform_numpy(X).flatten()
        return self

    def _fit_pandas(self, X):
        self.total_missing = self.missingness._transform_pandas(X).values.flatten()
        return self

    def _partial_fit_pandas(self, X):
        if self.total_missing is None:
            self.total_missing = self.missingness._transform_pandas(X).values.flatten()
        else:
            self.total_missing += self.missingness._transform_pandas(X).values.flatten()
        return self

    def _fit_polars(self, X):
        self.total_missing = self.missingness._transform_polars(X).to_numpy().flatten()
        return self

    def _partial_fit_polars(self, X):
        if self.total_missing is None:
            self.total_missing = (
                self.missingness._transform_polars(X).to_numpy().flatten()
            )
        else:
            self.total_missing += (
                self.missingness._transform_polars(X).to_numpy().flatten()
            )
        return self

    def _fit_arrow(self, X):
        self.total_missing = np.array(
            [v[0] for _, v in self.missingness._transform_arrow(X).to_pydict().items()]
        )
        return self

    def _partial_fit_arrow(self, X):
        if self.total_missing is None:
            total_missing = self.missingness._transform_arrow(X)
            self.total_missing = np.array(
                [v[0] for _, v in total_missing.to_pydict().items()]
            )
        else:
            other_missing = self.missingness._transform_arrow(X)
            other_missing = np.array(
                [v[0] for _, v in other_missing.to_pydict().items()]
            )
            self.total_missing += other_missing
        return self

    def _pool_fit(self, out: List["MinPrevalenceFeatureSelector"]):
        new_self = out[0]
        if len(out) > 1:
            logger.info("Pooling results")
            new_self.total_missing = np.sum(
                np.vstack([x.total_missing for x in out]), axis=0
            )
        logger.info(
            f"Total missing values: {new_self.total_missing.sum()} "
            f"({new_self.total_missing.mean():.2f} per feature)"
        )
        return new_self

    def _process_fit_output(self, input, out):
        self.config._feature_idx_out = _filter_features(
            self.height,
            self.total_missing,
            self.config.min_prevalence,
        )
        return super()._process_fit_output(input, out)
