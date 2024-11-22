"""
Correlation calculation
"""

from dataclasses import dataclass, field

from biocore import DataHandler

from biofit.integration.biosets import get_feature
from biofit.processing import (
    SelectedColumnTypes,
    SelectedFeatureTypes,
    sync_backup_config,
)
from biofit.utils import logging

from ..stat import Stat, StatConfig

logger = logging.get_logger(__name__)

CORRELATION_STAT_DOCSTRING = """
Stat features based on the correlation with the target variable.

Args:
    X: Input data
    y: Target variable
    input_columns: Columns to filter
    target_column: Target column
    **kwargs: Additional keyword arguments

Returns:
    SampleFiltered data.
"""


@dataclass
class CorrelationStatConfig(StatConfig):
    _transform_input_feature_types: SelectedFeatureTypes = field(
        default_factory=lambda: [None, get_feature("TARGET_FEATURE_TYPES")],
        init=False,
        repr=False,
    )
    _transform_unused_feature_types: SelectedFeatureTypes = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES"), None],
        init=False,
        repr=False,
    )
    processor_name: str = field(default="correlation", init=False, repr=False)

    input_columns: str = None
    target_column: str = None
    method: str = "auto"

    def __post_init__(self):
        if self.method not in [
            "auto",
            "pearsonr",
            "spearmanr",
            "kendalltau",
            "pointbiserialr",
        ]:
            raise ValueError(
                f"Method {self.method} not supported. Supported methods are: pearsonr, spearmanr, kendalltau, pointbiserialr"
            )
        if self.method != "auto":
            self._transform_process_desc = (
                f"Calculating {self.method} correlation with target variable"
            )


class CorrelationStat(Stat):
    """
    Correlation calculation based on the correlation with the target variable.
    """

    _config_class = CorrelationStatConfig
    config: CorrelationStatConfig
    output_dtype = "float64"
    func = None

    @sync_backup_config
    def set_params(self, **kwargs):
        self.config = self.config.replace_defaults(**kwargs)
        import scipy.stats

        if self.config.method != "auto":
            self.func = getattr(scipy.stats, self.config.method)
        return self

    def transform(
        self,
        X,
        y,
        input_columns: SelectedColumnTypes = None,
        target_column: SelectedColumnTypes = None,
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
            input_columns, target_column
        )
        return self._process_transform(
            X,
            y,
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
        raise_if_missing: bool = None,
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
            y,
            input_columns=input_columns,
            target_column=target_column,
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

    def _check_data(self, X, y):
        if self.config.method == "pointbiserialr":
            dtypes = next(iter(DataHandler.get_dtypes(y).values()))
            if not any(dtype in dtypes for dtype in ["int", "bool", "str"]):
                raise ValueError(
                    f"Point biserial correlation can only be used with binary target variables. Data type: {dtypes}"
                )
            n_unique = DataHandler.nunique(
                y, DataHandler.get_column_names(y, generate_cols=True)[0]
            )
            if n_unique != 2:
                raise ValueError(
                    f"Point biserial correlation can only be used with binary target variables. Number of unique values: {n_unique}"
                )
            return y, X  # the first argument must be the binary target variable
        return X, y

    def _process_transform_input(self, X, **kwargs):
        if self.config.method == "auto":
            import scipy.stats

            inds = kwargs["fn_kwargs"]["extra_indices"][0]
            is_categorical = DataHandler.is_categorical(X, inds, threshold=0.05)
            if is_categorical:
                n_classes = DataHandler.nunique(X, inds)
                if n_classes == 2:
                    self.config.method = "pointbiserialr"
                    kwargs["desc"] = (
                        "Calculating point biserial correlation with target variable"
                    )
                else:
                    self.config.method = "pearsonr"
                    kwargs["desc"] = (
                        "Calculating pearson correlation with target variable"
                    )
            else:
                self.config.method = "pearsonr"
                kwargs["desc"] = "Calculating pearson correlation with target variable"
            self.func = getattr(scipy.stats, self.config.method)
        return X, kwargs

    def _transform_sklearn(self, X, y):
        cols = DataHandler.get_column_names(X, generate_cols=True)
        corrs = {}
        for col in cols:
            corr, _ = self.func(
                *self._check_data(
                    DataHandler.select_column(X, col), DataHandler.select_column(y, 0)
                )
            )
            corrs[str(col)] = [corr]
        return corrs
