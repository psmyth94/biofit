from pathlib import Path
from typing import List, Union

import numpy as np
from biocore import DataHandler
from sklearn.pipeline import Pipeline

from biofit.metrics import calculate_metrics, confusion_matrix, get_metrics
from biofit.processing import BaseProcessor
from biofit.train_eval_utils import (
    _get_data,
    get_model_info,
    preprocess,
    save_confusion_matrix,
    save_metrics,
    save_predictions,
)


def _predict(
    models,
    x_eval,
    y_eval=None,
    preprocessors=None,
    use_proba=True,
    task=None,
    label_names=None,
    cache_dir=None,
):
    # model can be a list of models for each y_eval, if multi-label
    # classification/regression is used
    if not isinstance(models, list):
        models = [models]

    y_preds = []
    for i, model in enumerate(models):
        preprocessor = preprocessors[i] if preprocessors is not None else None
        if isinstance(model, Pipeline):
            if len(model.steps) > 1 and preprocessor is None:
                preprocessor = Pipeline([p for p in model.steps[:-1]])
            model = model.steps[-1][1]
        if preprocessor:
            x_eval, _, _, _ = preprocess(
                preprocessor, x_eval, cache_dir=cache_dir, transform_only=True
            )

        def fn(model, use_proba, x_test):
            extra_kwargs = {}
            if isinstance(model, BaseProcessor):
                extra_kwargs = {"cache_dir": cache_dir, "load_from_cache_file": False}

            if use_proba and hasattr(model, "predict_proba"):
                return model.predict_proba(x_test, **extra_kwargs)
            elif hasattr(model, "predict"):
                return model.predict(x_test, **extra_kwargs)
            else:
                return model.transform(x_test, **extra_kwargs)

        if not hasattr(model, "predict") and not hasattr(model, "predict_proba"):
            for m in model:
                y_pred = fn(m, use_proba, x_eval)
        elif hasattr(model, "steps"):
            for step in model.steps:
                y_pred = fn(
                    step[-1] if isinstance(step, tuple) else step, use_proba, x_eval
                )
        else:
            y_pred = fn(model, use_proba, x_eval)

        y_pred_dims = DataHandler.get_shape(y_pred)
        x_test_dims = DataHandler.get_shape(x_eval)
        if y_eval is not None:
            if task == "multiclass_classification":
                if label_names is None:
                    label_names = np.unique(y_eval).tolist()

                if len(y_pred_dims) == 1 and (
                    (y_pred_dims[0] % len(label_names)) != 0
                    or (y_pred_dims[0] // len(label_names)) != x_test_dims[0]
                ):
                    y_pred_ohe = np.zeros((x_test_dims[0], len(label_names)))
                    y_pred_ohe[np.arange(x_test_dims[0]), y_pred] = 1
                    y_pred = y_pred_ohe
                elif y_pred_dims[1] != len(label_names):
                    y_pred_ohe = np.zeros((y_pred_dims[0], len(label_names)))
                    y_pred_ohe[np.arange(y_pred_dims[0]), y_pred] = 1
                    y_pred = y_pred_ohe

            if task == "binary_classification":
                if len(y_pred_dims) == 1 and (
                    (y_pred_dims[0] % 2) != 0 or (y_pred_dims[0] // 2) != x_test_dims[0]
                ):
                    y_pred = DataHandler.concat([1 - y_pred, y_pred], axis=1)
                elif len(y_pred_dims) > 1 and y_pred_dims[1] != 2:
                    y_pred = DataHandler.concat([1 - y_pred, y_pred], axis=1)

        y_preds.append(y_pred)

    if len(y_preds) == 1:
        y_preds = y_preds[0]
    else:
        y_preds = DataHandler.concat(y_preds, axis=1)
    return y_preds


def _update_metrics(y_true, y_pred, labels, metrics, task, results):
    calc_results = calculate_metrics(metrics, y_true, y_pred, task, labels)
    results.update(calc_results)


def evaluate(
    model,
    data,
    target=None,
    label_names=None,
    input_columns: Union[List[str], str] = "auto",
    target_columns: Union[List[str], str] = "auto",
    preprocessors=None,
    task=None,
    metrics=None,
    use_proba=True,
    output_dir=None,
    cache_dir=None,
    config=None,
):
    """
    Evaluate the model or models on the test data with improved flexibility and error
    handling.

    Args:
        data (np.ndarray): Data to be used for testing the model.
        target (np.ndarray, *optional*): Labels corresponding to the test data.
        models (list): A list of models to evaluate.
        preprocessor (object, *optional*):
            Preprocessor to apply to the data before evaluation.
        metrics (dict):
            Metrics to calculate for evaluation, with each metric being a callable. if
            None, all metrics for the `task` will be calculated.
        use_proba (bool):
            Whether to use the predict_proba method of the model for predictions.
        save_indices (bool): Whether to save the indices of the test set to a file.
        output_dir (str): Directory to save the predictions table
        cache_dir (str):
            Directory to save the cache files. Defaults to f"{output_dir}/cache", if
            output_dir is provided.

    Returns:
        list: A list of dictionaries with metric results for each model.
    """

    x_eval, y_eval, _, _, _, _, input_columns, target_columns = _get_data(
        data=data,
        target=target,
        valid_data=None,
        valid_target=None,
        input_columns=input_columns,
        target_columns=target_columns,
        format=None,
        target_required=False,
    )

    model_info = get_model_info(model, task)
    if task is None:
        task = model_info.get("task", None)

    if (
        preprocessors is not None
        and not isinstance(preprocessors, list)
        and isinstance(model, list)
    ):
        preprocessors = [preprocessors] * len(model)

    if "classification" in task and label_names is None:
        label_names = DataHandler.to_list(DataHandler.unique(y_eval))

    if task and metrics is None and y_eval is not None:
        if task == "classification":
            if len(label_names) > 2:
                task = "multiclass_classification"
            else:
                task = "binary_classification"
        metrics = get_metrics(task)

    if (
        preprocessors is not None
        and not isinstance(preprocessors, list)
        and isinstance(model, list)
    ):
        preprocessors = [preprocessors] * len(model)

    y_preds = _predict(
        x_eval=x_eval,
        models=model,
        preprocessors=preprocessors,
        use_proba=use_proba,
    )
    y_preds = DataHandler.to_pandas(y_preds)

    results = None
    if y_eval is not None:
        results = calculate_metrics(
            metrics, DataHandler.to_pandas(y_eval), y_preds, task, label_names
        )

    y_preds = y_preds.sort_index()
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        y_preds = DataHandler.to_pandas(y_preds)
        labs = model_info.get("class_names", None) or label_names
        if results is not None:
            save_metrics(output_dir, results)

            cm = confusion_matrix(y_eval, y_preds, labels=labs)
            save_confusion_matrix(output_dir, cm, label_names)
        save_predictions(
            output_dir,
            y_preds,
            data,
            y_eval,
            label_names=label_names,
            target_columns=target_columns,
        )

    return y_preds, results


def predict(
    model,
    data,
    input_columns: Union[List[str], str] = "auto",
    preprocessors=None,
    use_proba=True,
    output_dir=None,
    cache_dir=None,
    config=None,
):
    """
    Evaluate the model or models on the test data with improved flexibility and error
    handling.

    Args:
        data (np.ndarray): Data to be used for testing the model.
        target (np.ndarray, *optional*): Labels corresponding to the test data.
        models (list): A list of models to evaluate.
        preprocessor (object, *optional*):
            Preprocessor to apply to the data before evaluation.
        use_proba (bool):
            Whether to use the predict_proba method of the model for predictions.
        output_dir (str): Directory to save the predictions table
        cache_dir (str):
            Directory to save the cache files. Defaults to f"{output_dir}/cache", if
            output_dir is provided.
        config (dict, *optional*):
            Placeholder for additional configuration options. Currently not used.

    Returns:
        list: A list of dictionaries with metric results for each model.
    """

    model_info = get_model_info(model, task=None)
    x_eval, _, _, _, _, _, input_columns, _ = _get_data(
        data=data,
        input_columns=input_columns,
        target_columns=None,
        format=None,
        target_required=False,
    )

    if (
        preprocessors is not None
        and not isinstance(preprocessors, list)
        and isinstance(model, list)
    ):
        preprocessors = [preprocessors] * len(model)

    y_preds = _predict(
        x_eval=x_eval,
        models=model,
        preprocessors=preprocessors,
        use_proba=use_proba,
    )
    y_preds = DataHandler.to_pandas(y_preds)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        y_pred = DataHandler.to_pandas(y_preds)
        save_predictions(
            output_dir,
            y_pred,
            data,
            label_names=model_info.get("class_names", None),
        )

    return y_preds
