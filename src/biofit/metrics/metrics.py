import copy
import inspect
from typing import Union

import numpy as np
import pandas as pd
from biocore import DataHandler
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
)
from sklearn.metrics import (
    log_loss,
    multilabel_confusion_matrix,
)
from sklearn.metrics._classification import (
    _check_set_wise_labels,
    _check_zero_division,
    _nanaverage,
    _prf_divide,
)
from sklearn.utils.class_weight import compute_class_weight


def confusion_matrix(
    y_true, y_pred, labels=None, sample_weight=None, samplewise=False, normalize=None
):
    y_true = DataHandler.to_numpy(y_true)
    y_pred = DataHandler.to_numpy(y_pred)

    if y_true.ndim == 1 or y_true.shape[1] == 1:
        y_true = y_true.flatten()
    if y_pred.ndim == 1 or y_pred.shape[1] == 1:
        y_pred = (y_pred.flatten() > 0.5).astype(int)
    else:
        y_pred = np.argmax(y_pred, axis=1)

    if len(labels) < 3:
        mat = sk_confusion_matrix(
            y_true,
            y_pred,
            labels=labels
            if labels is not None and not isinstance(labels[0], str)
            else None,
            sample_weight=sample_weight,
            normalize=normalize,
        )

        df = pd.DataFrame(
            mat,
            columns=[f"Predicted {label}" for label in labels],
            index=[f"Actual {label}" for label in labels],
        )
        return df
    else:
        # create a len(labels) x len(labels) matrix
        mat = np.zeros((len(labels), len(labels)), dtype=int)
        for i, label in enumerate(labels):
            for j, pred in enumerate(labels):
                mat[i, j] = np.sum((y_true == i) & (y_pred == j))

        df = pd.DataFrame(
            mat,
            columns=[f"Predicted {label}" for label in labels],
            index=[f"Actual {label}" for label in labels],
        )
        return df


def specificity(
    y_true,
    y_pred,
    *,
    labels=None,
    pos_label=1,
    average=None,
    sample_weight=None,
    zero_division="warn",
):
    _check_zero_division(zero_division)
    labels = _check_set_wise_labels(y_true, y_pred, average, labels, pos_label)

    # Calculate tp_sum, pred_sum, true_sum ###
    samplewise = average == "samples"
    MCM = multilabel_confusion_matrix(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        labels=labels,
        samplewise=samplewise,
    )
    tp_sum = MCM[:, 1, 1]
    tn_sum = MCM[:, 0, 0]
    fp_sum = MCM[:, 0, 1]
    fn_sum = MCM[:, 1, 0]
    true_sum = tp_sum + fn_sum
    false_sum = tn_sum + fp_sum

    if average == "micro":
        false_sum = np.array([false_sum.sum()])

    # Divide, and on zero-division, set scores and/or warn according to
    # zero_division:
    specificity = _prf_divide(
        tn_sum, false_sum, "specificity", "false", average, "specificity", zero_division
    )

    # Average the results
    if average == "weighted":
        weights = true_sum
    elif average == "samples":
        weights = sample_weight
    else:
        weights = None

    if average is not None:
        assert average != "binary" or len(specificity) == 1
        specificity = _nanaverage(specificity, weights=weights)
    if isinstance(specificity, (list, tuple, np.ndarray)) and len(specificity) > 1:
        specificity = specificity[-1]
    return specificity


def log_loss_weighted(y_true, y_pred, labels=None, class_weights=None):
    ytrue = DataHandler.to_numpy(y_true).flatten()
    if labels is None:
        labels = DataHandler.unique(DataHandler.to_numpy(ytrue).flatten())
    else:
        labels = DataHandler.to_numpy(labels).flatten()
        if set(labels) - set(ytrue):
            labels = DataHandler.unique(DataHandler.to_numpy(ytrue).flatten())

    if class_weights is None:
        class_weights = np.ones(len(labels))
    else:
        class_weights = compute_class_weight(class_weights, classes=labels, y=ytrue)

    ypred = DataHandler.to_numpy(y_pred)
    if ypred.ndim == 2 and ypred.shape[1] == 1:
        ypred = ypred.flatten()
    class_weights = dict(zip(labels, class_weights))
    sample_weights = np.array([class_weights[label] for label in ytrue])
    return log_loss(ytrue, ypred, sample_weight=sample_weights)


def calculate_metrics(
    metrics: Union[dict, callable], y_true, y_pred, sub_task, labels=None
):
    results = {}

    if labels is not None:
        labels = list(range(len(labels)))
    if callable(metrics):
        params = inspect.signature(metrics).parameters
        eval_kwargs = {}
        if "labels" in params:
            eval_kwargs["labels"] = labels

        results["custom_metric"].append(metrics(y_true, y_pred, **eval_kwargs))

    else:
        for metric_name, (metric_func, _) in metrics.items():
            if sub_task == "binary_classification":
                if metric_name in ["logloss", "logloss_weighted", "auc"]:
                    results[metric_name] = metric_func(
                        y_true, DataHandler.select_column(y_pred, 1)
                    )
                else:
                    results[metric_name] = metric_func(
                        y_true,
                        DataHandler.ge(DataHandler.select_column(y_pred, 1), 0.5),
                    )
            elif sub_task == "multiclass_classification":
                if metric_name in (
                    "accuracy",
                    "balanced_accuracy",
                    "f1_macro",
                    "f1_micro",
                    "f1_weighted",
                    "precision_macro",
                    "precision_micro",
                    "precision_weighted",
                    "recall_macro",
                    "recall_micro",
                    "recall_weighted",
                    "specificity_macro",
                    "specificity_micro",
                    "specificity_weighted",
                ):
                    results[metric_name] = metric_func(
                        y_true, DataHandler.argmax(y_pred, axis=1), labels
                    )
                else:
                    results[metric_name] = metric_func(y_true, y_pred, labels)
            else:
                if metric_name == "rmsle":
                    temp_pred = copy.deepcopy(DataHandler.to_numpy(y_pred))
                    temp_pred = np.clip(temp_pred, 0, None)
                    results[metric_name] = metric_func(y_true, temp_pred)
                else:
                    results[metric_name] = metric_func(y_true, y_pred)
    return results
