from collections import OrderedDict

from biofit.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping

from .configuration_auto import CONFIG_MAPPING_NAMES

MODEL_MAPPING_NAMES = OrderedDict(
    [
        ("xgboost", "XGBoostModel"),
        ("lightgbm", "LightGBMModel"),
        ("catboost", "CatBoostModel"),
        ("random_forest", "RandomForestModel"),
        ("logistic_regression", "LogisticRegressionModel"),
        ("lasso", "LassoModel"),
        ("svm", "SVMModel"),
        ("knn", "KNNModel"),
    ]
)

MODEL_FOR_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("xgboost", "XGBoostForClassification"),
        ("lightgbm", "LightGBMForClassification"),
        ("catboost", "CatBoostForClassification"),
        ("random_forest", "RandomForestForClassification"),
        ("logistic_regression", "LogisticRegressionForClassification"),
        ("lasso", "LassoForClassification"),
        ("svm", "SVMForClassification"),
        ("knn", "KNNForClassification"),
    ]
)

MODEL_FOR_REGRESSION_MAPPING_NAMES = OrderedDict(
    [
        ("xgboost", "XGBoostForRegression"),
        ("lightgbm", "LightGBMForRegression"),
        ("catboost", "CatBoostForRegression"),
        ("random_forest", "RandomForestForRegression"),
        ("logistic_regression", "LogisticRegressionForRegression"),
        ("lasso", "LassoForRegression"),
        ("svm", "SVMForRegression"),
        ("knn", "KNNForRegression"),
    ]
)

MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES)
MODEL_FOR_CLASSIFICATION_MAPPING_NAMES = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_CLASSIFICATION_MAPPING_NAMES
)


class AutoModel(_BaseAutoModelClass):
    _processor_mapping = MODEL_MAPPING


class AutoModelForClassification(_BaseAutoModelClass):
    _processor_mapping = MODEL_FOR_CLASSIFICATION_MAPPING_NAMES


class AutoModelForRegression(_BaseAutoModelClass):
    _processor_mapping = MODEL_FOR_REGRESSION_MAPPING_NAMES
