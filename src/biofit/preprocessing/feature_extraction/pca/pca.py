from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Union

import numpy as np
import pandas as pd
from biofit.processing import SelectedColumnTypes, sync_backup_config
from biofit.utils import logging
from sklearn.decomposition import PCA

from ..feature_extraction import FeatureExtractor, FeatureExtractorConfig

if TYPE_CHECKING:
    import polars as pl

logger = logging.get_logger(__name__)


@dataclass
class PCAFeatureExtractorConfig(FeatureExtractorConfig):
    _fit_process_desc: str = field(
        default="Determining the principal components", init=False, repr=False
    )
    _transform_process_desc: str = field(
        default="Transforming the input data to principal components",
        init=False,
        repr=False,
    )
    processor_name: str = field(default="pca", init=False, repr=False)

    input_columns: List[str] = None
    n_components: int = None
    copy: bool = True
    whiten: bool = False
    svd_solver: str = "auto"
    tol: float = 0.0
    iterated_power: str = "auto"
    n_oversamples: int = 10
    power_iteration_normalizer: str = "auto"
    random_state: int = None


class PCAFeatureExtractor(FeatureExtractor):
    output_dtype = "float64"
    _config_class = PCAFeatureExtractorConfig
    config: PCAFeatureExtractorConfig

    def __init__(
        self,
        input_columns: List[str] = None,
        n_components: int = None,
        copy: bool = True,
        whiten: bool = False,
        svd_solver: str = "auto",
        tol: float = 0.0,
        iterated_power: str = "auto",
        n_oversamples: int = 10,
        power_iteration_normalizer: str = "auto",
        random_state: int = None,
        config: PCAFeatureExtractorConfig = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            input_columns=input_columns,
            n_components=n_components,
            copy=copy,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            n_oversamples=n_oversamples,
            power_iteration_normalizer=power_iteration_normalizer,
            random_state=random_state,
            **kwargs,
        )
        self.pca = PCA(
            n_components=self.config.n_components,
            copy=self.config.copy,
            whiten=self.config.whiten,
            svd_solver=self.config.svd_solver,
            tol=self.config.tol,
            iterated_power=self.config.iterated_power,
            n_oversamples=self.config.n_oversamples,
            power_iteration_normalizer=self.config.power_iteration_normalizer,
            random_state=self.config.random_state,
        )

    @sync_backup_config
    def set_params(self, **kwargs):
        self.config = self.config.replace_defaults(**kwargs)
        self.pca = PCA(
            n_components=self.config.n_components,
            copy=self.config.copy,
            whiten=self.config.whiten,
            svd_solver=self.config.svd_solver,
            tol=self.config.tol,
            iterated_power=self.config.iterated_power,
            n_oversamples=self.config.n_oversamples,
            power_iteration_normalizer=self.config.power_iteration_normalizer,
            random_state=self.config.random_state,
        )
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
    ) -> "PCAFeatureExtractor":
        self.config._input_columns = self._set_input_columns_and_arity(
            input_columns or self.config.input_columns
        )
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
        self._input_columns = self._set_input_columns_and_arity(
            input_columns or self.config.input_columns
        )
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

    def _fit_sklearn(self, X: Union[pd.DataFrame, "pl.DataFrame", np.ndarray]):
        self.config.estimator = self.pca.fit(X)
        return self

    def _process_fit_output(self, input, out):
        self.config._n_features_out = self.config.estimator.n_components_
        return super()._process_fit_output(input, out)

    def _transform_sklearn(self, X: Union[pd.DataFrame, "pl.DataFrame", np.ndarray]):
        return self.config.estimator.transform(X)
