from dataclasses import dataclass, field
from functools import wraps

import pyarrow as pa
from biocore import DataHandler
from biocore.utils.import_util import is_datasets_available

from biofit.processing import BaseProcessor, ProcessorConfig, SelectedColumnTypes
from biofit.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class ModelConfig(ProcessorConfig):
    """Base class for feature extraction processor configurations."""

    _fit_process_desc: str = field(default="Fitting the model", init=False, repr=False)
    predict_process_desc: str = field(
        default="Predicting target output", init=False, repr=False
    )
    predict_proba_process_desc: str = field(
        default="Predicting target output probabilities", init=False, repr=False
    )
    processor_type: str = field(default="models", init=False, repr=False)
    _missing_val: float = field(default=0, init=False, repr=False)
    _missing_val_pa_type: pa.DataType = field(
        default=pa.float64(), init=False, repr=False
    )
    class_names: list = field(default=None, init=False, repr=False)

    task: str = None


class Model(BaseProcessor):
    """Base class for models."""

    @wraps(BaseProcessor._process_transform)
    def predict(
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
        """Predict the model."""
        self._method_prefix = "_predict"
        self.output_dtype = "int64"
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

    def predict_proba(
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
        self._method_prefix = "_predict_proba"
        self.output_dtype = "float64"
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

    @staticmethod
    def suggest_params() -> dict:
        """Get hyperparameters for an optuna trial."""
        raise NotImplementedError

    def _process_fit_input(self, X, **kwargs):
        target_col = kwargs["fn_kwargs"]["extra_indices"][0]
        if target_col and "class" in self.config.task.lower():
            if isinstance(target_col, list):
                target_col = target_col[0]

            target_col = DataHandler.get_column_names(X, generate_cols=True)[target_col]

            if self.config.n_classes is None:
                self.config.n_classes = DataHandler.nunique(X, target_col)

            if is_datasets_available():
                from datasets import Dataset

                if isinstance(X, Dataset):
                    feat = X._info.features[target_col]
                    self.config.class_names = feat.names

            if self.config.n_classes > 2:
                self.config.task = "multiclass_classification"
            else:
                self.config.task = "binary_classification"
        return super()._process_fit_input(X, **kwargs)

    def _process_fit_output(self, input, out):
        if hasattr(self, "classes_"):
            if self.config.class_names is not None and isinstance(
                self.classes_[0], int
            ):
                if sorted(self.classes_.tolist()) == list(range(self.config.n_classes)):
                    self.config.class_names = [
                        self.config.class_names[i] for i in self.classes_
                    ]
                else:
                    self.config.class_names = [str(i) for i in self.classes_]
            else:
                self.config.class_names = [str(i) for i in self.classes_]
        return super()._process_fit_output(input, out)

    def _process_transform_input(self, X, **kwargs):
        if self._method_prefix == "_predict_proba":
            self.config._n_features_out = self.config.n_classes
            self.config._feature_names_out = self.config.class_names
            self.output_dtype = "float64"
        else:
            if self.config.extra_names_in_ is not None:
                self.config._feature_names_out = self.config.extra_names_in_[0]
            self.config._n_features_out = 1
        return super()._process_transform_input(X, **kwargs)

    def _process_transform_batch_input(self, X, *fn_args, **fn_kwargs):
        X, args, kwargs = super()._process_transform_batch_input(
            X, *fn_args, **fn_kwargs
        )
        if DataHandler.supports_named_columns(X):
            input_cols = DataHandler.get_column_names(X)
            missing_cols = set(self.config.feature_names_in_) - set(input_cols)
            intersecting_cols = set(input_cols) & set(self.config.feature_names_in_)

            X_dims = DataHandler.get_shape(X)
            num_cols = None
            if len(X_dims) > 1:
                num_cols = X_dims[1]

            num_non_existing = num_cols - len(intersecting_cols)
            if num_non_existing:
                logger.warning_once(
                    f"Dataset has {num_non_existing} out of {num_cols} columns that were "
                    "not in the training data. Dropping these columns."
                )
                X = DataHandler.select_columns(X, list(intersecting_cols))

            if missing_cols:
                if self.config._missing_val is None:
                    self.config._missing_val_str = "`None`"
                elif self.config._missing_val == 0:
                    self.config._missing_val_str = "zeroes"
                else:
                    self.config._missing_val_str = f"{self.config._missing_val}"
                logger.warning_once(
                    f"Dataset is missing {len(missing_cols)} out of "
                    f"{len(self.config.feature_names_in_)} columns that were in the "
                    "training data. Adding these columns as "
                    f"{self.config._missing_val_str}."
                )
                num_rows = DataHandler.get_shape(X)[0]
                zeros_mat = pa.table(
                    {
                        col: pa.array(
                            [self.config._missing_val] * num_rows,
                            type=self.config._missing_val_pa_type,
                        )
                        for col in missing_cols
                    }
                )
                X = DataHandler.concat(
                    [X, DataHandler.to_format(zeros_mat, DataHandler.get_format(X))],
                    axis=1,
                )
            # Reorder columns to match the training data
            X = DataHandler.select_columns(X, self.config.feature_names_in_)

        return X, args, kwargs
