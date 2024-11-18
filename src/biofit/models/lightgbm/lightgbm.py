from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Type, Union

import numpy as np
from biocore import DataHandler
from biocore.utils.import_util import is_lightgbm_available, requires_backends

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes, sync_backup_config
from biofit.utils import logging

from ..models import Model, ModelConfig

logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from lightgbm import LGBMClassifier, LGBMRegressor
    from lightgbm.sklearn import _LGBM_ScikitCustomObjectiveFunction

if is_lightgbm_available():
    from lightgbm import LGBMClassifier, LGBMRegressor
    from lightgbm.sklearn import _LGBM_ScikitCustomObjectiveFunction


@dataclass
class LightGBMConfig(ModelConfig):
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [
            None,
            get_feature("TARGET_FEATURE_TYPES"),
            None,
            None,
        ],
        init=False,
        repr=False,
    )
    _fit_unused_feature_types: List[Type] = field(
        default_factory=lambda: [
            get_feature("METADATA_FEATURE_TYPES"),
            None,
            None,
            None,
        ],
        init=False,
        repr=False,
    )
    _transform_unused_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES")],
        init=False,
        repr=False,
    )
    processor_name: str = field(default="lightgbm", init=False, repr=False)
    boosting_type: str = "gbdt"
    num_leaves: int = 31
    max_depth: int = -1
    learning_rate: float = 0.1
    n_estimators: int = 100
    subsample_for_bin: int = 200000
    objective: Optional[Union[str, "_LGBM_ScikitCustomObjectiveFunction"]] = None
    eval_metric: str = None
    class_weight: Optional[Union[Dict, str]] = None
    min_split_gain: float = 0.0
    min_child_weight: float = 1e-3
    min_child_samples: int = 20
    subsample: float = 1.0
    subsample_freq: int = 0
    colsample_bytree: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    random_state: Optional[Union[int, np.random.RandomState]] = 42
    n_jobs: Optional[int] = None
    importance_type: str = "split"
    early_stopping_rounds: int = None
    callbacks: Optional[List[Callable]] = None
    verbosity: int = -1
    additional_kwargs: dict = field(default_factory=dict)

    n_classes: int = None
    task: str = None
    estimator: Union["LGBMClassifier", "LGBMRegressor"] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        requires_backends(self.__class__, "lightgbm")
        self._fit_process_desc = f"Fitting the LGBM {self.task} model"
        self.predict_process_desc = f"Predicting with the LGBM {self.task} model"
        self.predict_proba_process_desc = (
            f"Predicting probabilities with the LGBM {self.task} model"
        )


class LightGBMModel(Model):
    config_class = LightGBMConfig
    config: LightGBMConfig
    lightgbm: Union["LGBMClassifier", "LGBMRegressor"]

    def __init__(
        self,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        objective: Optional[Union[str, "_LGBM_ScikitCustomObjectiveFunction"]] = None,
        eval_metric: str = None,
        class_weight: Optional[Union[Dict, str]] = None,
        min_split_gain: float = 0.0,
        min_child_weight: float = 1e-3,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        n_jobs: Optional[int] = None,
        importance_type: str = "split",
        callbacks: Optional[List[Callable]] = None,
        verbosity: int = -1,
        n_classes: int = None,
        early_stopping_rounds: int = None,
        task: str = None,
        version: str = None,
        config: Optional[LightGBMConfig] = None,
        **kwargs,
    ):
        m_params = [f.name for f in fields(LightGBMConfig) if f.init]
        biofit_params = {k: v for k, v in kwargs.items() if k in m_params}
        lightgbm_params = {k: v for k, v in kwargs.items() if k not in m_params}
        if "additional_kwargs" in biofit_params:
            lightgbm_params.update(biofit_params.pop("additional_kwargs"))
        super().__init__(
            config=config,
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample_for_bin=subsample_for_bin,
            objective=objective,
            eval_metric=eval_metric,
            class_weight=class_weight,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            importance_type=importance_type,
            early_stopping_rounds=early_stopping_rounds,
            callbacks=callbacks,
            verbosity=verbosity,
            n_classes=n_classes,
            task=task,
            version=version,
            additional_kwargs=lightgbm_params,
            **biofit_params,
        )
        if self.config.task is None:
            raise ValueError(
                "Task is not set. Please set the task before setting the parameters."
            )

        if "classification" in self.config.task:
            self.lightgbm = LGBMClassifier(
                boosting_type=self.config.boosting_type,
                num_leaves=self.config.num_leaves,
                max_depth=self.config.max_depth if self.config.max_depth > 0 else -1,
                learning_rate=self.config.learning_rate,
                n_estimators=self.config.n_estimators,
                subsample_for_bin=self.config.subsample_for_bin,
                objective=self.config.objective,
                class_weight=self.config.class_weight
                if self.config.class_weight != "None"
                else None,
                min_split_gain=self.config.min_split_gain,
                min_child_weight=self.config.min_child_weight,
                min_child_samples=self.config.min_child_samples,
                subsample=self.config.subsample,
                subsample_freq=self.config.subsample_freq,
                colsample_bytree=self.config.colsample_bytree,
                reg_alpha=self.config.reg_alpha,
                reg_lambda=self.config.reg_lambda,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                importance_type=self.config.importance_type,
                verbosity=self.config.verbosity,
                **self.config.additional_kwargs,
            )
        elif "regression" in self.config.task:
            self.lightgbm = LGBMRegressor(
                boosting_type=self.config.boosting_type,
                num_leaves=self.config.num_leaves,
                max_depth=self.config.max_depth if self.config.max_depth > 0 else -1,
                learning_rate=self.config.learning_rate,
                n_estimators=self.config.n_estimators,
                subsample_for_bin=self.config.subsample_for_bin,
                objective=self.config.objective,
                min_split_gain=self.config.min_split_gain,
                min_child_weight=self.config.min_child_weight,
                min_child_samples=self.config.min_child_samples,
                subsample=self.config.subsample,
                subsample_freq=self.config.subsample_freq,
                colsample_bytree=self.config.colsample_bytree,
                reg_alpha=self.config.reg_alpha,
                reg_lambda=self.config.reg_lambda,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                importance_type=self.config.importance_type,
                verbosity=self.config.verbosity,
                **self.config.additional_kwargs,
            )
        else:
            raise ValueError(f"Invalid task: {self.config.task}")

    @sync_backup_config
    def set_params(self, **params):
        self.config = self.config.replace_defaults(**params)
        if self.config.task is None:
            raise ValueError(
                "Task is not set. Please set the task before setting the parameters."
            )

        if "classification" in self.config.task:
            self.lightgbm = LGBMClassifier(
                boosting_type=self.config.boosting_type,
                num_leaves=self.config.num_leaves,
                max_depth=self.config.max_depth if self.config.max_depth > 0 else -1,
                learning_rate=self.config.learning_rate,
                n_estimators=self.config.n_estimators,
                subsample_for_bin=self.config.subsample_for_bin,
                objective=self.config.objective,
                class_weight=self.config.class_weight
                if self.config.class_weight != "None"
                else None,
                min_split_gain=self.config.min_split_gain,
                min_child_weight=self.config.min_child_weight,
                min_child_samples=self.config.min_child_samples,
                subsample=self.config.subsample,
                subsample_freq=self.config.subsample_freq,
                colsample_bytree=self.config.colsample_bytree,
                reg_alpha=self.config.reg_alpha,
                reg_lambda=self.config.reg_lambda,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                importance_type=self.config.importance_type,
                verbosity=self.config.verbosity,
            )
        elif "regression" in self.config.task:
            self.lightgbm = LGBMRegressor(
                boosting_type=self.config.boosting_type,
                num_leaves=self.config.num_leaves,
                max_depth=self.config.max_depth if self.config.max_depth > 0 else -1,
                learning_rate=self.config.learning_rate,
                n_estimators=self.config.n_estimators,
                subsample_for_bin=self.config.subsample_for_bin,
                objective=self.config.objective,
                min_split_gain=self.config.min_split_gain,
                min_child_weight=self.config.min_child_weight,
                min_child_samples=self.config.min_child_samples,
                subsample=self.config.subsample,
                subsample_freq=self.config.subsample_freq,
                colsample_bytree=self.config.colsample_bytree,
                reg_alpha=self.config.reg_alpha,
                reg_lambda=self.config.reg_lambda,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                importance_type=self.config.importance_type,
                verbosity=self.config.verbosity,
            )
        else:
            raise ValueError(f"Invalid task: {self.config.task}")
        return self

    def set_objective(self, task: str):
        if task == "regression":
            if self.config.eval_metric == "rmse":
                self.config.objective = "regression"
            elif self.config.eval_metric == "mae":
                self.config.objective = "regression_l1"
        elif task == "binary_classification":
            self.config.objective = "binary"
            if self.config.eval_metric == "accuracy":
                self.config.eval_metric = "binary_error"
        elif task == "multiclass_classification":
            self.config.objective = "multiclass"
        self.lightgbm = self.lightgbm.set_params(objective=self.config.objective)
        return self

    @property
    def feature_importances_(self):
        return self.config.estimator.feature_importances_

    @property
    def feature_names_in_(self):
        return self.config.feature_names_in_

    def fit(
        self,
        X,
        y=None,
        eval_set=None,
        input_columns: SelectedColumnTypes = None,
        target_column: SelectedColumnTypes = None,
        eval_input_columns: SelectedColumnTypes = None,
        eval_target_column: SelectedColumnTypes = None,
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
        early_stopping_rounds: int = None,
    ) -> "LightGBMModel":
        self.config._input_columns = self._set_input_columns_and_arity(
            input_columns, target_column, eval_input_columns, eval_target_column
        )
        if eval_set is not None:
            if isinstance(eval_set, tuple):
                extras = eval_set
            elif (
                isinstance(eval_set, list)
                and len(eval_set) == 1
                and isinstance(eval_set[0], tuple)
            ):
                extras = eval_set[0]
            else:
                raise ValueError(
                    "eval_set must be a tuple or a list containing a single tuple"
                )
        else:
            extras = (None, None)

        return self._process_fit(
            X,
            y,
            *extras,
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
        self._method_prefix = "_predict"
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

    def _fit_sklearn(self, X, y, eval_x=None, eval_y=None):
        callbacks = self.config.callbacks
        if self.config.early_stopping_rounds:
            callbacks = callbacks or []
            from lightgbm.callback import early_stopping

            callbacks.append(early_stopping(self.config.early_stopping_rounds))

        if eval_x is not None and eval_y is not None:
            self.config.estimator = self.lightgbm.fit(
                X,
                y,
                eval_set=[(eval_x, eval_y)],
                eval_metric=self.config.eval_metric,
                callbacks=callbacks,
            )
        else:
            self.config.estimator = self.lightgbm.fit(
                X, y, eval_metric=self.config.eval_metric, callbacks=callbacks
            )
        if self.config.early_stopping_rounds:
            self.config.n_estimators = self.config.estimator.best_iteration_
            self.config.estimator.set_params(n_estimators=self.config.n_estimators)
            self.config.early_stopping_rounds = None
        return self

    def _predict_sklearn(self, X):
        return self.config.estimator.predict(X)

    def _predict_proba_sklearn(self, X):
        return self.config.estimator.predict_proba(X)

    def _get_features_out(
        self, X, selected_indices=None, unselected_indices=None, **kwargs
    ):
        if self._method_prefix == "_predict_proba":
            self.config._n_features_out = self.config.n_classes
            self.output_dtype = "float64"
        else:
            self.config._n_features_out = 1
        return super()._get_features_out(
            X,
            selected_indices=selected_indices,
            unselected_indices=unselected_indices,
            one_to_one_features=False,
            n_features_out=self.config._n_features_out,
            keep_unused_columns=False,
        )

    @staticmethod
    def suggest_params(data):
        params = {
            "verbosity": -1,
            "n_estimators": ("suggest_int", [50, 1000], {"step": 50}),
            "max_depth": ("suggest_int", [0, 11], {}),
            "learning_rate": ("suggest_float", [1e-3, 0.1], {"log": True}),
            "num_leaves": ("suggest_int", [31, 255], {}),
            "subsample": ("suggest_float", [0.5, 1.0], {"step": 0.1}),
            "reg_alpha": ("suggest_float", [1e-9, 1.0], {"log": True}),
            "reg_lambda": ("suggest_float", [1e-9, 1.0], {"log": True}),
            "min_child_samples": ("suggest_int", [10, 100], {}),
            "min_split_gain": ("suggest_float", [0.0, 1.0], {}),
            "num_threads": -1,
        }

        if data is not None:
            data_dim = DataHandler.get_shape(data)

        if data_dim[1] > data_dim[0]:
            msg = "The number of features is greater than the number of samples"
            max_feat_frac = max(0.5, data_dim[0] / data_dim[1])
            max_feat_frac = round(max_feat_frac, 1)
            params["colsample_bytree"] = (
                "suggest_float",
                [0.1, max_feat_frac],
                {"step": 0.1},
            )
            if data_dim[1] > 1000:
                msg += " and exceeds 1000"
                params["n_estimators"] = ("suggest_int", [50, 500], {"step": 50})
                params["max_depth"] = ("suggest_int", [3, 8], {})
                params["num_leaves"] = ("suggest_int", [31, 63], {})
                params["min_child_samples"] = ("suggest_int", [10, 50], {})
                params["max_bin"] = ("suggest_int", [63, 127], {})
            msg += ". Adjusting the hyperparameters accordingly for faster training. "
            logger.warning(msg)

        return params

    @staticmethod
    def suggest_first_trial():
        return {
            "n_estimators": 100,
            "max_depth": 8,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "subsample": 1.0,
            "reg_alpha": 1e-9,
            "reg_lambda": 1e-9,
            "min_child_samples": 20,
            "colsample_bytree": 0.9,
            "min_split_gain": 0.0,
        }


class LightGBMForClassification(LightGBMModel):
    def __init__(self, **kwargs):
        kwargs["task"] = kwargs.get("task", "classification")
        super().__init__(**kwargs)

    @property
    def classes_(self):
        return self.config.estimator.classes_

    @sync_backup_config
    def set_params(self, **params):
        self.config.task = self.config.task or "classification"
        return super().set_params(**params)

    def _process_fit_input(self, X, **kwargs):
        X, kwargs = super()._process_fit_input(X, **kwargs)
        self.set_objective(self.config.task)
        return X, kwargs

    @staticmethod
    def suggest_params(data):
        params = LightGBMModel.suggest_params(data)
        params["class_weight"] = ("suggest_categorical", [["balanced", "None"]], {})
        return params

    @staticmethod
    def suggest_first_trial():
        return {
            **LightGBMModel.suggest_first_trial(),
            "class_weight": "balanced",
        }


class LightGBMForRegression(LightGBMModel):
    @sync_backup_config
    def set_params(self, **params):
        self.config.task = self.config.task or "regression"
        return super().set_params(**params)
