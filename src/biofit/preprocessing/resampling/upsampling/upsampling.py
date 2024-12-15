from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Type, Union

import numpy as np
from biocore import DataHandler
from biocore.utils.import_util import is_imblearn_available, requires_backends

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes, sync_backup_config
from biofit.utils import logging

from ..resampling import Resampler, ResamplerConfig

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

    from biofit.processing import BaseProcessor

logger = logging.get_logger(__name__)

OVER_SAMPLING_METHODS = {}

if is_imblearn_available():
    from imblearn.over_sampling import (
        ADASYN,
        SMOTE,
        SMOTEN,
        SMOTENC,
        SVMSMOTE,
        KMeansSMOTE,
        RandomOverSampler,
    )

    OVER_SAMPLING_METHODS = {
        "random": RandomOverSampler,
        "smote": SMOTE,
        "smoten": SMOTEN,
        "smotenc": SMOTENC,
        "svmsmote": SVMSMOTE,
        "kmeanssmote": KMeansSMOTE,
        "adasyn": ADASYN,
    }


@dataclass
class UpSamplerConfig(ResamplerConfig):
    # process descriptions
    processor_name: str = field(default="upsampling", init=False, repr=False)
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [None, get_feature("TARGET_FEATURE_TYPES")],
        init=False,
        repr=False,
    )
    _fit_unused_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES"), None],
        init=False,
        repr=False,
    )
    _transform_unused_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES")],
        init=False,
        repr=False,
    )

    input_columns: List[str] = field(default=None, init=False, repr=False)
    target_column: str = field(default=None, init=False, repr=False)
    method: str = "random"

    sampling_strategy: str = "auto"
    random_state: int = None

    # RandomOverSampler specific attributes
    shrinkage: dict = None

    # BaseSMOTE attributes
    k_neighbors = 5
    n_jobs = None

    # SMOTEN specific attributes
    categorical_encoder: Union["BaseEstimator", "BaseProcessor"] = None

    # SMOTENC specific attributes
    categorical_features: list = None

    # SVMSMOTE specific attributes
    m_neighbors: int = 10
    svm_estimator: Union["BaseEstimator", "BaseProcessor"] = None
    out_step: float = 0.5

    # KMeansSMOTE specific attributes
    kmeans_estimator: Union[int, "BaseEstimator", "BaseProcessor"] = None
    density_exponent: Union[float, str] = "auto"
    cluster_balance_threshold: Union[float, str] = "auto"

    # ADASYN specific attributes
    n_neighbors: int = 5

    def __post_init__(self):
        requires_backends("upsampling", "imblearn")
        if self.random_state is None:
            self.random_state = np.random.randint(0, np.iinfo(np.int32).max)
        if self.method not in OVER_SAMPLING_METHODS:
            raise ValueError(
                f"Invalid method {self.method}. Valid methods are {list(OVER_SAMPLING_METHODS.keys())}"
            )

        self.sampler_kwargs_ = {}
        self.sampler_kwargs_["sampling_strategy"] = self.sampling_strategy
        self.sampler_kwargs_["random_state"] = self.random_state

        if self.method == "random":
            self.sampler_kwargs_["sampling_strategy"] = self.sampling_strategy

        if self.method in [
            "smote",
            "smoten",
            "smotenc",
            "svmsmote",
            "kmeanssmote",
            "adasyn",
        ]:
            self.sampler_kwargs_["n_jobs"] = self.n_jobs

        if self.method in ["smote", "smoten", "smotenc", "svmsmote", "kmeanssmote"]:
            self.sampler_kwargs_["k_neighbors"] = self.k_neighbors

        if self.method in ["smoten", "smotenc"]:
            self.sampler_kwargs_["categorical_encoder"] = self.categorical_encoder

        if self.method == "smotenc":
            self.sampler_kwargs_["categorical_features"] = self.categorical_features

        if self.method == "svmsmote":
            self.sampler_kwargs_["m_neighbors"] = self.m_neighbors
            self.sampler_kwargs_["svm_estimator"] = self.svm_estimator
            self.sampler_kwargs_["out_step"] = self.out_step

        if self.method == "kmeanssmote":
            self.sampler_kwargs_["kmeans_estimator"] = self.kmeans_estimator
            self.sampler_kwargs_["density_exponent"] = self.density_exponent
            self.sampler_kwargs_["cluster_balance_threshold"] = (
                self.cluster_balance_threshold
            )

        if self.method == "adasyn":
            self.sampler_kwargs_["n_neighbors"] = self.n_neighbors


@dataclass
class UpSamplerConfigForMetagenomics(UpSamplerConfig):
    # dataset specific attributes
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            (get_feature("Abundance"), get_feature("ReadCount")),
            get_feature("TARGET_FEATURE_TYPES"),
        ],
        init=False,
        repr=False,
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [(get_feature("Abundance"), get_feature("ReadCount"))],
        init=False,
        repr=False,
    )
    experiment_name: str = field(default="metagenomics", init=False, repr=False)


@dataclass
class UpSamplerConfigForOTU(UpSamplerConfig):
    # dataset specific attributes
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("Abundance"),
            get_feature("TARGET_FEATURE_TYPES"),
        ],
        init=False,
        repr=False,
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
    experiment_name: str = field(default="otu", init=False, repr=False)


@dataclass
class UpSamplerConfigForSNP(UpSamplerConfig):
    # dataset specific attributes
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("GenomicVariant"),
            get_feature("TARGET_FEATURE_TYPES"),
        ],
        init=False,
        repr=False,
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("GenomicVariant")], init=False, repr=False
    )
    experiment_name: str = field(default="snp", init=False, repr=False)


class UpSampler(Resampler):
    output_dtype = "float32"

    # config class
    _config_class = UpSamplerConfig
    config: UpSamplerConfig

    def __init__(
        self,
        method: str = "random",
        sampling_strategy: str = "auto",
        random_state: int = None,
        shrinkage: dict = None,
        k_neighbors=5,
        n_jobs=None,
        categorical_encoder: Union["BaseEstimator", "BaseProcessor"] = None,
        categorical_features: list = None,
        m_neighbors: int = 10,
        svm_estimator: Union["BaseEstimator", "BaseProcessor"] = None,
        out_step: float = 0.5,
        kmeans_estimator: Union[int, "BaseEstimator", "BaseProcessor"] = None,
        density_exponent: Union[float, str] = "auto",
        cluster_balance_threshold: Union[float, str] = "auto",
        n_neighbors: int = 5,
        config: Union[
            UpSamplerConfig,
            UpSamplerConfigForOTU,
            UpSamplerConfigForSNP,
            UpSamplerConfigForMetagenomics,
        ] = None,
        **kwargs,
    ):
        """
        Upsample the minority class(es) in the dataset.

        Args:
            method (str, 'random'):
                The method to use for oversampling. Possible options are:
                - 'random': Random Over Sampler
                - 'smote': Synthetic Minority Over-sampling Technique
                - 'smoten': SMOTE for numerical features only
                - 'smotenc': SMOTE for numerical and categorical features
                - 'svmsmote': SVM-SMOTE
                - 'kmeanssmote': KMeans-SMOTE
                - 'adasyn': Adaptive Synthetic Sampling Approach for Imbalanced Learning

            sampling_strategy (str, 'auto'):
                The sampling strategy to use for oversampling. Possible options are:
                - 'auto': Automatically resample the minority class(es) to the majority class(es) size.
                - 'all': Resample all classes to the same size.

            random_state (int, *optional*):
                The random state to use for sampling.

            shrinkage (dict, *optional*):
                The shrinkage parameter for RandomOverSampler.

            k_neighbors (int, 5):
                The number of nearest neighbors to use for SMOTE, SMOTEN, SMOTENC, SVMSMOTE, and KMeansSMOTE.

            n_jobs (int, *optional*):
                The number of jobs to use for SMOTE, SMOTEN, SMOTENC, SVMSMOTE, and KMeansSMOTE.

            categorical_encoder (Union[BaseEstimator, BaseProcessor], *optional*):
                The encoder to use for SMOTEN.

            categorical_features (list, *optional*):
                The list of categorical features to use for SMOTENC.

            m_neighbors (int, 10):
                The number of nearest neighbors to use for SVMSMOTE.

            svm_estimator (Union[BaseEstimator, BaseProcessor], *optional*):
                The estimator to use for SVMSMOTE.

            out_step (float, 0.5):
                The outlier step to use for SVMSMOTE.

            kmeans_estimator (Union[int, BaseEstimator, BaseProcessor], *optional*):
                The estimator to use for KMeansSMOTE.

            density_exponent (Union[float, str], 'auto'):
                The exponent to use for KMeansSMOTE.

            cluster_balance_threshold (Union[float, str], 'auto'):
                The balance threshold to use for KMeansSMOTE.

            n_neighbors (int, 5):
                The number of nearest neighbors to use for ADASYN.

            config (Union[UpSamplerConfig, UpSamplerConfigForOTU, UpSamplerConfigForSNP, UpSamplerConfigForMetagenomics], *optional*):
                The configuration to use for the upsampling process. If provided, the other arguments are ignored.
                Use set_params to update the configuration after initialization.
            **kwargs:
                Additional keyword arguments to be passed to ProcessorConfig
        """
        super().__init__(
            config=config,
            method=method,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            shrinkage=shrinkage,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
            categorical_encoder=categorical_encoder,
            categorical_features=categorical_features,
            m_neighbors=m_neighbors,
            svm_estimator=svm_estimator,
            out_step=out_step,
            kmeans_estimator=kmeans_estimator,
            density_exponent=density_exponent,
            cluster_balance_threshold=cluster_balance_threshold,
            n_neighbors=n_neighbors,
            **kwargs,
        )

        self.over_sampler = OVER_SAMPLING_METHODS[self.config.method](
            **self.config.sampler_kwargs_
        )

    @sync_backup_config
    def set_params(self, **kwargs):
        self.config = self.config.replace_defaults(**kwargs)
        self.over_sampler = OVER_SAMPLING_METHODS[self.config.method](
            **self.config.sampler_kwargs_
        )
        return self

    def fit(
        self,
        X,
        y=None,
        input_columns: SelectedColumnTypes = None,
        target_column: SelectedColumnTypes = None,
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
    ) -> "UpSampler":
        self.config._input_columns = self._set_input_columns_and_arity(
            input_columns, target_column
        )
        return self._process_fit(
            X,
            y,
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
        y=None,
        input_columns: SelectedColumnTypes = None,
        target_column: SelectedColumnTypes = None,
        keep_unused_columns: bool = True,
        raise_if_missing: bool = True,
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
            y,
            input_columns=input_columns,
            target_column=target_column,
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

    def _fit_pandas(self, X, y):
        self.over_sampler.fit_resample(X, DataHandler.select_column(y, 0))
        self.config.sample_indices = self.over_sampler.sample_indices_
        return self

    def _fit_numpy(self, X, y):
        self.over_sampler.fit_resample(X, DataHandler.select_column(y, 0))
        self.config.sample_indices = self.over_sampler.sample_indices_
        return self

    def _process_transform_input(self, X, **kwargs):
        batch_size = kwargs.get("batch_size", None)
        if batch_size is not None and batch_size < DataHandler.get_shape(X)[0]:
            logger.warning(
                "Upsampling does not support batch processing. Ignoring batched and batch_size parameters."
            )
            kwargs["batch_size"] = None
        self.config.fingerprint = kwargs.get("new_fingerprint", None)
        kwargs["features"] = None
        return X, kwargs

    def _transform_any(self, X):
        return {"indices": self.config.sample_indices.flatten()}
