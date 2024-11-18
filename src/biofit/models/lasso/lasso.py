from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
from sklearn.linear_model import Lasso

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes, sync_backup_config
from biofit.utils import logging

from ..logistic_regression import LogisticRegressionModel
from ..models import Model, ModelConfig

logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    pass


@dataclass
class LassoConfig(ModelConfig):
    _fit_input_feature_types: List[Union[None, type]] = field(
        default_factory=lambda: [None, get_feature("TARGET_FEATURE_TYPES")],
        init=False,
        repr=False,
    )
    _fit_unused_feature_types: List[Union[None, type]] = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES"), None],
        init=False,
        repr=False,
    )
    _transform_unused_feature_types: List[type] = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES")],
        init=False,
        repr=False,
    )
    _fit_process_desc: str = field(
        default="Fitting the lasso model", init=False, repr=False
    )
    predict_process_desc: str = field(
        default=None,
        init=False,
        repr=False,
    )
    predict_proba_process_desc: str = field(
        default=None,
        init=False,
        repr=False,
    )
    processor_name: str = field(default="lasso", init=False, repr=False)

    input_columns: SelectedColumnTypes = None
    target_column: SelectedColumnTypes = None
    alpha = 1.0
    fit_intercept: bool = True
    precompute: Union[bool, np.ndarray] = False
    copy_X: bool = True
    max_iter: int = 1000
    tol: float = 1e-4
    warm_start: bool = False
    positive: bool = False
    random_state: Optional[int] = 42
    class_weight: Union[dict, str] = None
    selection: str = "cyclic"  # "cyclic" or "random"
    solver: str = "saga"

    use_predict_proba: bool = False
    task: str = None
    estimator: Lasso = field(default=None, init=False, repr=False)
    n_classes: int = None

    def __post_init__(self):
        if self.task is None:
            self.task = "classification" if self.use_predict_proba else "regression"
        self._fit_process_desc = f"Fitting the lasso {self.task} model"
        self.predict_process_desc = f"Predicting with the lasso {self.task} model"
        self.predict_proba_process_desc = (
            f"Predicting probabilities with the lasso {self.task} model"
        )


class LassoConfigForOTU(LassoConfig):
    dataset_name: str = field(default="otu", init=False, repr=False)
    log_transform: Union[str, bool] = field(default="log2_1p", init=False, repr=False)


class LassoModel(Model):
    config_class = LassoConfig
    config: LassoConfig
    lasso: Lasso

    @sync_backup_config
    def set_params(self, **kwargs):
        self.config = self.config.replace_defaults(**kwargs)
        self.lasso = Lasso(
            alpha=self.config.alpha,
            fit_intercept=self.config.fit_intercept,
            precompute=self.config.precompute,
            copy_X=self.config.copy_X,
            max_iter=self.config.max_iter,
            tol=self.config.tol,
            warm_start=self.config.warm_start,
            positive=self.config.positive,
            random_state=self.config.random_state,
            selection=self.config.selection,
        )
        return self

    @property
    def feature_importances_(self):
        return self.config.estimator.coef_.flatten()

    @property
    def feature_names_in_(self):
        return self.config.feature_names_in_

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
    ) -> "LassoModel":
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

    def _fit_sklearn(self, X, y):
        if self.config.transform_log == "log2_1p":
            X = np.log2(X + 1)
        self.config.estimator = self.lasso.fit(X, y)
        return self

    def _predict_sklearn(self, X):
        return self.config.estimator.predict(X)

    def _predict_proba_sklearn(self, X):
        logger.warning_once(
            "predict_proba is not supported for Lasso. Returning using predict instead."
        )
        return self._predict_sklearn(X)

    @staticmethod
    def suggest_params(data):
        return {
            "alpha": (
                "suggest_float",
                (
                    1e-8,
                    1e3,
                ),
                {"log": True},
            ),
            "fit_intercept": ("suggest_categorical", ([True, False],), {}),
            "max_iter": (
                "suggest_int",
                (
                    1000,
                    10000,
                ),
                {},
            ),
        }

    @staticmethod
    def suggest_first_trial():
        return {
            "alpha": 1.0,
            "fit_intercept": True,
            "max_iter": 1000,
        }


# Lasso for Classification is a logistic regression model with L1 penalty, except
# that with the additional layer of converting the scores for classes to the "winning"
# class output label.
class LassoForClassification(LogisticRegressionModel):
    class_config = LassoConfig

    def __init__(
        self,
        alpha=1.0,
        fit_intercept: bool = True,
        copy_X: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        warm_start: bool = False,
        class_weight: str = None,
        random_state: int = 42,
        solver: str = "saga",
        config: LassoConfig = None,
        **kwargs,
    ):
        if "C" in kwargs:
            alpha = 1.0 / kwargs.pop("C")
        super().__init__(
            C=1.0 / alpha,  # C is the inverse of alpha
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            penalty="l1",
            solver=solver,
            warm_start=warm_start,
            class_weight=class_weight,
            random_state=random_state,
            config=config,
        )

    @sync_backup_config
    def set_params(self, **kwargs):
        if "alpha" in kwargs:
            kwargs["C"] = 1.0 / kwargs.pop("alpha")
        super().set_params(**kwargs)
        return self

    @property
    def classes_(self):
        return self.config.estimator.classes_

    @staticmethod
    def suggest_params(data):
        params = {
            "C": (
                "suggest_float",
                (
                    1e-4,
                    1e2,
                ),
                {"log": True},
            ),
            "tol": (
                "suggest_float",
                (
                    1e-5,
                    1e-2,
                ),
                {"log": True},
            ),
            "fit_intercept": ("suggest_categorical", ([True, False],), {}),
            "max_iter": ("suggest_categorical", ([100, 200, 500, 1000],), {}),
            "class_weight": ("suggest_categorical", (["balanced", "None"],), {}),
        }
        return params

    @staticmethod
    def suggest_first_trial():
        return {
            "alpha": 1.0,
            "tol": 1e-4,
            "fit_intercept": True,
            "max_iter": 1000,
            "class_weight": "None",
        }


class LassoForRegression(LassoModel):
    pass
