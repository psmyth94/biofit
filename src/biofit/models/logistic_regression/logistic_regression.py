from dataclasses import dataclass, field
from typing import List, Optional, Union

from sklearn.linear_model import LogisticRegression

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes, sync_backup_config
from biofit.utils import logging

from ..models import Model, ModelConfig

logger = logging.get_logger(__name__)


@dataclass
class LogisticRegressionConfig(ModelConfig):
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
    processor_name: str = field(default="logistic_regression", init=False, repr=False)

    input_columns: SelectedColumnTypes = None
    target_column: SelectedColumnTypes = None

    penalty: str = "l2"
    dual: bool = False
    tol: float = 1e-4
    C: float = 1.0
    fit_intercept: bool = True
    intercept_scaling: float = 1
    class_weight: Union[dict, str] = None
    random_state: Optional[int] = 42
    solver: str = "saga"
    max_iter: int = 100
    multi_class: str = "auto"
    verbose: int = 0
    warm_start: bool = False
    n_jobs: Optional[int] = None
    l1_ratio: Optional[float] = None

    use_predict_proba: bool = False
    task: str = field(default="classification", init=False, repr=False)
    estimator: LogisticRegression = field(default=None, init=False, repr=False)
    n_classes: int = None

    def __post_init__(self):
        self._fit_process_desc = "Fitting the logistic regression model"
        self.predict_process_desc = "Predicting with the logistic regression model"
        self.predict_proba_process_desc = (
            "Predicting probabilities with the logistic regression model"
        )


class LogisticRegressionModel(Model):
    config_class = LogisticRegressionConfig
    config: LogisticRegressionConfig
    logistic_regression: LogisticRegression

    def __init__(
        self,
        penalty: str = "l2",
        dual: bool = False,
        tol: float = 1e-4,
        C: float = 1.0,
        fit_intercept: bool = True,
        intercept_scaling: float = 1,
        class_weight: Union[dict, str] = None,
        random_state: int = 42,
        solver: str = "lbfgs",
        max_iter: int = 100,
        multi_class: str = "auto",
        verbose: int = 0,
        warm_start: bool = False,
        n_jobs: Optional[int] = None,
        l1_ratio: Optional[float] = None,
        config: LogisticRegressionConfig = None,
        **kwargs,
    ):
        super().__init__(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class if multi_class != "None" else None,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio,
            config=config,
            **kwargs,
        )
        self.logistic_regression = LogisticRegression(
            penalty=self.config.penalty,
            dual=self.config.dual,
            tol=self.config.tol,
            C=self.config.C,
            fit_intercept=self.config.fit_intercept,
            intercept_scaling=self.config.intercept_scaling,
            class_weight=self.config.class_weight
            if self.config.class_weight != "None"
            else None,
            random_state=self.config.random_state,
            solver=self.config.solver,
            max_iter=self.config.max_iter,
            multi_class=self.config.multi_class,
            verbose=self.config.verbose,
            warm_start=self.config.warm_start,
            n_jobs=self.config.n_jobs,
            l1_ratio=self.config.l1_ratio,
        )

    @sync_backup_config
    def set_params(self, **kwargs):
        self.config = self.config.replace_defaults(**kwargs)
        self.logistic_regression = LogisticRegression(
            penalty=self.config.penalty,
            dual=self.config.dual,
            tol=self.config.tol,
            C=self.config.C,
            fit_intercept=self.config.fit_intercept,
            intercept_scaling=self.config.intercept_scaling,
            class_weight=self.config.class_weight
            if self.config.class_weight != "None"
            else None,
            random_state=self.config.random_state,
            solver=self.config.solver,
            max_iter=self.config.max_iter,
            multi_class=self.config.multi_class,
            verbose=self.config.verbose,
            warm_start=self.config.warm_start,
            n_jobs=self.config.n_jobs,
            l1_ratio=self.config.l1_ratio,
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
    ) -> "LogisticRegressionModel":
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

    def _fit_sklearn(self, X, y):
        self.config.estimator = self.logistic_regression.fit(X, y)
        return self

    def _predict_sklearn(self, X):
        return self.config.estimator.predict(X)

    def _predict_proba_sklearn(self, X):
        return self.config.estimator.predict_proba(X)

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
            "max_iter": ("suggest_categorical", ([100, 200, 500, 1000],), {}),
            "penalty": "l1",
            "n_jobs": -1,
            "class_weight": ("suggest_categorical", (["balanced", "None"],), {}),
        }
        return params
