from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Type

import numpy as np
from biocore import DataHandler
from sklearn.metrics import DistanceMetric

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes, sync_backup_config

from ..stat import Stat, StatConfig

if TYPE_CHECKING:
    pass


@dataclass
class DistanceStatConfig(StatConfig):
    """
    A base class for distance metrics.

    Inherits all attributes and methods from StatConfig.

    This class is tailored to handle distance metrics, including Minkowski,
    weighted and unweighted, seuclidean, and mahalanobis.
    """

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
    processor_name: str = field(default="distance", init=False, repr=False)
    metric: str = "euclidean"


@dataclass
class DistanceStatConfigForMetagenomics(DistanceStatConfig):
    """
    Calculates distance metrics specifically for metagenomics data.

    Inherits all configs from DistanceStatConfig.

    This class is tailored to handle metagenomics data, including OTU abundance,
    ASV abundance, and read counts, using the 'braycurtis' metric by default.
    """

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
    metric: str = "braycurtis"


@dataclass
class DistanceStatConfigForOTU(DistanceStatConfig):
    """
    A subclass for calculating distance metrics on OTU abundance data.

    Inherits all configs from DistanceStatConfig, but is specifically
    tailored for OTU abundance data, defaulting to the 'braycurtis' metric.
    """

    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
    dataset_name: str = field(default="otu", init=False, repr=False)
    metric: str = "braycurtis"


@dataclass
class DistanceStatConfigForASV(DistanceStatConfig):
    """
    A subclass for calculating distance metrics on ASV abundance data.

    Inherits all attributes and methods from DistanceStat, but is specifically
    tailored for ASV abundance data, defaulting to the 'braycurtis' metric.
    """

    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("Abundance")], init=False, repr=False
    )
    dataset_name: str = field(default="asv", init=False, repr=False)
    metric: str = "braycurtis"


class DistanceStatConfigForReadCount(DistanceStatConfig):
    """
    A subclass for calculating distance metrics on read count data.

    Inherits all attributes and methods from DistanceStat, but is specifically
    tailored for read count data, defaulting to the 'braycurtis' metric.
    """

    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("ReadCount")], init=False, repr=False
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("ReadCount")], init=False, repr=False
    )
    dataset_name: str = field(default="read_count", init=False, repr=False)
    metric: str = "braycurtis"


class DistanceStatConfigForSNP(DistanceStatConfig):
    """
    A subclass for calculating distance metrics on read count data.

    Inherits all attributes and methods from DistanceStat, but is specifically
    tailored for read count data, defaulting to the 'braycurtis' metric.
    """

    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("GenomicVariant")], init=False, repr=False
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("GenomicVariant")], init=False, repr=False
    )
    dataset_name: str = field(default="snp", init=False, repr=False)

    metric: str = "jaccard"


class DistanceStat(Stat):
    """
    A base class for calculating distance metrics on genomic data.

    Attributes:
        metric (str): The name of the distance metric to use. Defaults to 'euclidean'.
        p (float): The p-norm to apply for Minkowski, weighted and unweighted. Default is 2.
        w (Union[None, np.ndarray], optional): The weight vector for weighted Minkowski. Default is None.
        V (Union[None, np.ndarray], optional): The variance vector for seuclidean. Default is None.
        VI (Union[None, np.ndarray], optional): The inverse of the covariance matrix for mahalanobis. Default is None.

    Methods:
        fit_sklearn(X: Union[np.ndarray, pd.DataFrame, "pl.DataFrame"]): Validates input data and calculates the pairwise distances between samples in X.
    """

    _config_class = DistanceStatConfig
    config: DistanceStatConfig
    output_dtype = "float64"

    def __init__(
        self,
        config: Optional[DistanceStatConfig] = None,
        metric: Optional[str] = "euclidean",
        **kwargs,
    ):
        super().__init__(config=config, metric=metric, **kwargs)
        self.pdist = DistanceMetric.get_metric(self.config.metric).pairwise

    @sync_backup_config
    def set_params(self, **kwargs):
        self.config = self.config.replace_defaults(**kwargs)
        self.pdist = DistanceMetric.get_metric(self.config.metric).pairwise
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
    ) -> "DistanceStat":
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
        self.config._n_features_out = DataHandler.get_shape(X)[0]
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

    def _transform_numpy(self, X: np.ndarray):
        return self.pdist(X, X)
