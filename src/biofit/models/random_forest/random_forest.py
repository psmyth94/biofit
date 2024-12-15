from dataclasses import dataclass, field
from typing import List, Union

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes, sync_backup_config

from ..models import Model, ModelConfig


@dataclass
class RandomForestConfig(ModelConfig):
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
        default="Fitting the random forest model", init=False, repr=False
    )
    predict_proba_process_desc: str = field(
        default=None,
        init=False,
        repr=False,
    )
    processor_name: str = field(default="random_forest", init=False, repr=False)

    n_estimators: int = 100
    max_depth: int = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_weight_fraction_leaf: float = 0.0
    max_leaf_nodes: int = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = False
    oob_score: bool = False
    n_jobs: int = None
    random_state: int = 42
    verbose: int = 0
    warm_start: bool = False
    ccp_alpha: float = 0.0
    max_samples: int = None
    monotonic_cst: dict = None
    criterion: str = None  # gini for classification, squared_error for regression
    max_features: Union[str, int, float] = "sqrt"

    class_weight: Union[str, dict, List[dict]] = None
    n_classes: int = None

    use_predict_proba: bool = False
    task: str = None

    estimator: Union[RandomForestClassifier, RandomForestRegressor] = field(
        default=None, init=False, repr=False
    )


class RandomForestModel(Model):
    _config_class = RandomForestConfig
    config: RandomForestConfig
    random_forest: Union[RandomForestClassifier, RandomForestRegressor]

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_leaf_nodes: int = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int = None,
        random_state: int = None,
        verbose: int = 0,
        warm_start: bool = False,
        ccp_alpha: float = 0.0,
        max_samples: int = None,
        monotonic_cst: dict = None,
        criterion: str = None,
        max_features: Union[str, int, float] = "sqrt",
        class_weight: Union[str, dict, List[dict]] = None,
        task: str = None,
        config: RandomForestConfig = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            monotonic_cst=monotonic_cst,
            criterion=criterion,
            max_features=max_features,
            task=task,
            class_weight=class_weight,
            **kwargs,
        )

        if self.config.task == "classification":
            self.random_forest = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                criterion=self.config.criterion if self.config.criterion else "gini",
                max_depth=self.config.max_depth if self.config.max_depth else None,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                min_weight_fraction_leaf=self.config.min_weight_fraction_leaf,
                max_features=self.config.max_features,
                max_leaf_nodes=self.config.max_leaf_nodes,
                min_impurity_decrease=self.config.min_impurity_decrease,
                bootstrap=self.config.bootstrap,
                oob_score=self.config.oob_score,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
                verbose=self.config.verbose,
                warm_start=self.config.warm_start,
                class_weight=self.config.class_weight
                if self.config.class_weight and self.config.class_weight != "None"
                else None,
                ccp_alpha=self.config.ccp_alpha,
                max_samples=self.config.max_samples,
                monotonic_cst=self.config.monotonic_cst,
            )
        else:
            self.random_forest = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                criterion=self.config.criterion
                if self.config.criterion
                else "squared_error",
                max_depth=self.config.max_depth if self.config.max_depth > 0 else None,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                min_weight_fraction_leaf=self.config.min_weight_fraction_leaf,
                max_features=self.config.max_features,
                max_leaf_nodes=self.config.max_leaf_nodes,
                min_impurity_decrease=self.config.min_impurity_decrease,
                bootstrap=self.config.bootstrap,
                oob_score=self.config.oob_score,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
                verbose=self.config.verbose,
                warm_start=self.config.warm_start,
                ccp_alpha=self.config.ccp_alpha,
                max_samples=self.config.max_samples,
                monotonic_cst=self.config.monotonic_cst,
            )

    @sync_backup_config
    def set_params(self, **kwargs):
        self.config = self.config.replace_defaults(**kwargs)
        if self.config.task == "classification":
            self.random_forest = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                criterion=self.config.criterion if self.config.criterion else "gini",
                max_depth=self.config.max_depth if self.config.max_depth else None,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                min_weight_fraction_leaf=self.config.min_weight_fraction_leaf,
                max_features=self.config.max_features,
                max_leaf_nodes=self.config.max_leaf_nodes,
                min_impurity_decrease=self.config.min_impurity_decrease,
                bootstrap=self.config.bootstrap,
                oob_score=self.config.oob_score,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
                verbose=self.config.verbose,
                warm_start=self.config.warm_start,
                class_weight=self.config.class_weight
                if self.config.class_weight and self.config.class_weight != "None"
                else None,
                ccp_alpha=self.config.ccp_alpha,
                max_samples=self.config.max_samples,
                monotonic_cst=self.config.monotonic_cst,
            )
        else:
            self.random_forest = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                criterion=self.config.criterion
                if self.config.criterion
                else "squared_error",
                max_depth=self.config.max_depth if self.config.max_depth > 0 else None,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                min_weight_fraction_leaf=self.config.min_weight_fraction_leaf,
                max_features=self.config.max_features,
                max_leaf_nodes=self.config.max_leaf_nodes,
                min_impurity_decrease=self.config.min_impurity_decrease,
                bootstrap=self.config.bootstrap,
                oob_score=self.config.oob_score,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
                verbose=self.config.verbose,
                warm_start=self.config.warm_start,
                ccp_alpha=self.config.ccp_alpha,
                max_samples=self.config.max_samples,
                monotonic_cst=self.config.monotonic_cst,
            )
        if self.config.use_predict_proba:
            self.output_dtype = "float32"
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
    ) -> "RandomForestModel":
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

    def _fit_sklearn(self, X, y):
        self.config.estimator = self.random_forest.fit(X, y)
        return self

    def _predict_sklearn(self, X):
        return self.config.estimator.predict(X)

    def _predict_proba_sklearn(self, X):
        return self.config.estimator.predict_proba(X)

    @staticmethod
    def suggest_params(data):
        params = {
            "n_estimators": ("suggest_int", (100, 2000), {}),
            "max_depth": ("suggest_int", (0, 11), {}),
            "min_samples_split": ("suggest_int", (2, 20), {}),
            "min_samples_leaf": ("suggest_int", (1, 20), {}),
            # "bootstrap": ("suggest_categorical", ([True, False],), {}),
            # "max_features": ("suggest_categorical", (["sqrt", "log2"],), {}),
            "n_jobs": -1,
        }

        return params

    @staticmethod
    def suggest_first_trial():
        return {
            "n_estimators": 100,
            "max_depth": 0,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "n_jobs": -1,
        }


class RandomForestForClassification(RandomForestModel):
    """Random Forest model for classification tasks"""

    @property
    def classes_(self):
        return self.config.estimator.classes_

    @sync_backup_config
    def set_params(self, **kwargs):
        kwargs.pop("task", None)
        self.config.task = "classification"
        return super().set_params(**kwargs)

    @staticmethod
    def suggest_params(data):
        params = RandomForestModel.suggest_params(data)
        params["criterion"] = ("suggest_categorical", (["gini", "entropy"],), {})
        params["class_weight"] = (
            "suggest_categorical",
            (["balanced", "balanced_subsample", "None"],),
            {},
        )
        return params

    @staticmethod
    def suggest_first_trial():
        return {
            **RandomForestModel.suggest_first_trial(),
            "criterion": "gini",
            "class_weight": "balanced",
        }


class RandomForestForRegression(RandomForestModel):
    @sync_backup_config
    def set_params(self, **kwargs):
        kwargs.pop("task", None)
        self.config.task = "regression"
        return super().set_params(**kwargs)

    @staticmethod
    def suggest_params(data):
        params = RandomForestModel.suggest_params(data)
        params["criterion"] = (
            "suggest_categorical",
            (["squared_error", "absolute_error", "poisson"],),
            {},
        )
        return params
