from functools import partial

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .metrics import (
    calculate_metrics,  # noqa: F401
    confusion_matrix,  # noqa: F401
    log_loss_weighted,
    specificity,
)

_OPTUNA_METRICS = {
    "binary_classification": "f1",
    "multiclass_classification": "f1_weighted",
    "regression": "rmse",
    "multi_regression": "rmse",
}

_METRICS = {
    "binary_classification": {
        "logloss": (log_loss, "minimize"),
        "logloss_weighted": (
            partial(log_loss_weighted, class_weights="balanced"),
            "minimize",
        ),
        "auc": (roc_auc_score, "maximize"),
        "f1": (f1_score, "maximize"),
        "accuracy": (accuracy_score, "maximize"),
        "balanced_accuracy": (balanced_accuracy_score, "maximize"),
        "precision": (precision_score, "maximize"),
        "recall": (recall_score, "maximize"),
        "specificity": (specificity, "maximize"),
    },
    "multiclass_classification": {
        "mlogloss": (
            lambda y_true, y_pred, labels: log_loss(y_true, y_pred, labels=labels),
            "minimize",
        ),
        "mlogloss_weighted": (
            lambda y_true, y_pred, labels: log_loss_weighted(
                y_true, y_pred, labels=labels, class_weights="balanced"
            ),
            "minimize",
        ),
        "accuracy": (
            lambda y_true, y_pred, labels: accuracy_score(y_true, y_pred),
            "maximize",
        ),
        "balanced_accuracy": (
            lambda y_true, y_pred, labels: balanced_accuracy_score(y_true, y_pred),
            "maximize",
        ),
        "f1_macro": (
            lambda y_true, y_pred, labels: f1_score(
                y_true, y_pred, average="macro", labels=labels
            ),
            "maximize",
        ),
        "f1_micro": (
            lambda y_true, y_pred, labels: f1_score(
                y_true, y_pred, average="micro", labels=labels
            ),
            "maximize",
        ),
        "f1_weighted": (
            lambda y_true, y_pred, labels: f1_score(
                y_true, y_pred, average="weighted", labels=labels
            ),
            "maximize",
        ),
        "precision_macro": (
            lambda y_true, y_pred, labels: precision_score(
                y_true, y_pred, average="macro", labels=labels
            ),
            "maximize",
        ),
        "precision_micro": (
            lambda y_true, y_pred, labels: precision_score(
                y_true, y_pred, average="micro", labels=labels
            ),
            "maximize",
        ),
        "precision_weighted": (
            lambda y_true, y_pred, labels: precision_score(
                y_true, y_pred, average="weighted", labels=labels
            ),
            "maximize",
        ),
        "recall_macro": (
            lambda y_true, y_pred, labels: recall_score(
                y_true, y_pred, average="macro", labels=labels
            ),
            "maximize",
        ),
        "recall_micro": (
            lambda y_true, y_pred, labels: recall_score(
                y_true, y_pred, average="micro", labels=labels
            ),
            "maximize",
        ),
        "recall_weighted": (
            lambda y_true, y_pred, labels: recall_score(
                y_true, y_pred, average="weighted", labels=labels
            ),
            "maximize",
        ),
        "specificity_macro": (
            lambda y_true, y_pred, labels: specificity(
                y_true, y_pred, average="macro", labels=labels
            ),
            "maximize",
        ),
        # "specificity_micro": (
        #     lambda y_true, y_pred, labels: specificity(
        #         y_true, y_pred, average="micro", labels=labels
        #     ),
        #     "maximize",
        # ),
        "specificity_weighted": (
            lambda y_true, y_pred, labels: specificity(
                y_true, y_pred, average="weighted", labels=labels
            ),
            "maximize",
        ),
    },
    "regression": {
        "rmse": (partial(mean_squared_error, squared=False), "minimize"),
        "rmsle": (partial(mean_squared_log_error, squared=False), "minimize"),
        "r2": (r2_score, "maximize"),
        "mse": (mean_squared_error, "minimize"),
        "mae": (mean_absolute_error, "minimize"),
    },
    "multi_regression": {
        "rmse": (partial(mean_squared_error, squared=False), "minimize"),
        "rmsle": (partial(mean_squared_log_error, squared=False), "minimize"),
        "r2": (r2_score, "maximize"),
        "mse": (mean_squared_error, "minimize"),
        "mae": (mean_absolute_error, "minimize"),
    },
    "multilabel_classification": {
        "logloss": (log_loss, "minimize"),
    },
}


def get_metrics(task=None, metric=None):
    if task is None:
        return _METRICS
    if metric is None:
        return _METRICS[task]
    else:
        return _METRICS[task][metric]


def get_optuna_metric(task):
    return _OPTUNA_METRICS[task]
