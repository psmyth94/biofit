import copy
import os
import sys
from dataclasses import asdict
from typing import TYPE_CHECKING, Callable, Generator, List, Union

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import yaml
from biocore.data_handling import DataHandler
from biocore.utils.import_util import (
    is_catboost_available,
    is_lightgbm_available,
    is_plotly_available,
    is_xgboost_available,
)
from biocore.utils.inspect import get_kwargs
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR

from biofit.auto.processing_auto import ProcessorPipeline
from biofit.models import (
    LightGBMForClassification,
    LightGBMForRegression,
    RandomForestForClassification,
    RandomForestForRegression,
)
from biofit.models.lasso.lasso import LassoForClassification, LassoForRegression
from biofit.models.models import Model
from biofit.processing import BaseProcessor
from biofit.utils import logging
from biocore.utils.py_util import is_bioset, is_dataset
from biofit.visualization.plotting_utils import plot_feature_importance

if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)


CLASSIFICATION_TASKS = ["binary_classification", "multiclass_classification"]
REGRESSION_TASKS = ["regression", "multi_regression"]

_MODELS = {
    "binary_classification": {
        "random_forest": RandomForestForClassification,
        "lightgbm": LightGBMForClassification,
        "svm": SVC,
        "gradient_boosting": GradientBoostingClassifier,
        "lasso": LassoForClassification,
    },
    "multiclass_classification": {
        "random_forest": RandomForestForClassification,
        "lightgbm": LightGBMForClassification,
        "svm": SVC,
        "gradient_boosting": GradientBoostingClassifier,
        "lasso": LassoForClassification,
    },
    "regression": {
        "random_forest": RandomForestForRegression,
        "lightgbm": LightGBMForRegression,
        "svm": SVR,
        "gradient_boosting": GradientBoostingRegressor,
        "lasso": LassoForRegression,
    },
    "multi_regression": {
        "random_forest": RandomForestForRegression,
        "lightgbm": LightGBMForRegression,
        "svm": SVR,
        "gradient_boosting": GradientBoostingRegressor,
        "lasso": LassoForRegression,
    },
}

if is_xgboost_available():
    from xgboost import XGBClassifier, XGBRegressor

    _MODELS["binary_classification"]["xgboost"] = XGBClassifier
    _MODELS["multiclass_classification"]["xgboost"] = XGBClassifier
    _MODELS["regression"]["xgboost"] = XGBRegressor
    _MODELS["multi_regression"]["xgboost"] = XGBRegressor

if is_lightgbm_available():
    _MODELS["binary_classification"]["lightgbm"] = LightGBMForClassification
    _MODELS["multiclass_classification"]["lightgbm"] = LightGBMForClassification
    _MODELS["regression"]["lightgbm"] = LightGBMForRegression
    _MODELS["multi_regression"]["lightgbm"] = LightGBMForRegression

if is_catboost_available():
    from catboost import CatBoostClassifier, CatBoostRegressor

    _MODELS["binary_classification"]["catboost"] = CatBoostClassifier
    _MODELS["multiclass_classification"]["catboost"] = CatBoostClassifier
    _MODELS["regression"]["catboost"] = CatBoostRegressor
    _MODELS["multi_regression"]["catboost"] = CatBoostRegressor


_CV = {
    "binary_classification": {
        "stratified_kfold": StratifiedKFold,
        "kfold": KFold,
        "group_kfold": GroupKFold,
        "stratified_group_kfold": StratifiedGroupKFold,
        "logo": LeaveOneGroupOut,
    },
    "multiclass_classification": {
        "stratified_kfold": StratifiedKFold,
        "kfold": KFold,
        "group_kfold": GroupKFold,
        "stratified_group_kfold": StratifiedGroupKFold,
        "logo": LeaveOneGroupOut,
    },
    "regression": {
        "stratified_kfold": StratifiedKFold,
        "kfold": KFold,
    },
    "multi_regression": {
        "stratified_kfold": StratifiedKFold,
        "kfold": KFold,
    },
    "multilabel_classification": {
        "stratified_kfold": StratifiedKFold,
        "kfold": KFold,
    },
}


def _flatten_pipeline(p):
    pipe = []
    if isinstance(p, list):
        for step in p:
            pipe.extend(_flatten_pipeline(step))
    elif isinstance(p, (Pipeline, ProcessorPipeline)):
        for step in p.steps:
            pipe.extend(
                _flatten_pipeline(step[-1] if isinstance(step, tuple) else step)
            )
    else:
        pipe.append(p)
    return pipe


def _get_name_and_params(obj):
    obj_params = None
    obj_name = None
    if obj:
        if isinstance(obj, (ProcessorPipeline, Pipeline, list)):
            if isinstance(obj, (Pipeline, ProcessorPipeline)):
                obj = [step[-1] if isinstance(step, tuple) else step for step in obj]
            obj_params = [
                p.get_params() if isinstance(p, BaseProcessor) else p.__dict__
                for p in obj
            ]
            obj_name = [
                p.config.processor_name
                if isinstance(p, BaseProcessor)
                and getattr(p.config, "processor_name", None)
                else p.__class__.__name__
                for p in obj
            ]
        else:
            obj_params = obj.get_params()
            obj_name = (
                obj.config.processor_name
                if getattr(obj.config, "processor_name", None)
                else obj.__class__.__name__
            )
    return obj_name, obj_params


def _get_processor_info(
    models,
    preprocessors=None,
):
    if isinstance(models, list) and len(models) == 1:
        models = models[0]

    preprocessor_name, preprocessor_params = None, None
    new_preprocessors = preprocessors
    if isinstance(models, Pipeline) and preprocessors is None:
        if len(models.steps) > 1:
            new_preprocessors = _flatten_pipeline(
                [p[-1] if isinstance(p, tuple) else p for p in models.steps[:-1]]
            )
        if isinstance(new_preprocessors, list):
            if len(new_preprocessors) == 1:
                new_preprocessors = new_preprocessors[0]
            else:
                new_preprocessors = ProcessorPipeline(new_preprocessors)

        new_models = (
            models.steps[-1][-1]
            if isinstance(models.steps[-1], tuple)
            else models.steps[-1]
        )
        model_name, model_params = _get_name_and_params(new_models)
        if new_preprocessors is not None:
            preprocessor_name, preprocessor_params = _get_name_and_params(
                new_preprocessors
            )
    elif isinstance(models, list):
        new_models, model_name, model_params = [], [], []
        new_preprocessors, preprocessor_name, preprocessor_params = [], [], []
        for i, m in enumerate(models):
            _preprocessors = None
            if isinstance(preprocessors, list):
                _preprocessors = preprocessors[i]
            if _preprocessors is None:
                if isinstance(m, Pipeline):
                    if len(m.steps) > 1:
                        _preprocessors = _flatten_pipeline(
                            [p[-1] if isinstance(p, tuple) else p for p in m.steps[:-1]]
                        )
                    m = (
                        m.steps[-1][-1]
                        if isinstance(m.steps[-1], tuple)
                        else m.steps[-1]
                    )
            if isinstance(_preprocessors, list):
                if len(_preprocessors) == 1:
                    _preprocessors = _preprocessors[0]
                else:
                    _preprocessors = ProcessorPipeline(_preprocessors)

            _name, _params = _get_name_and_params(m)
            new_models.append(m)
            model_name.append(_name)
            model_params.append(_params)
            if _preprocessors:
                _name, _params = _get_name_and_params(_preprocessors)
                preprocessor_name.append(_name)
                preprocessor_params.append(_params)
                new_preprocessors.append(_preprocessors)
        if len(preprocessor_name) == 0 or len(preprocessor_params) == 0:
            preprocessor_name, preprocessor_params = None, None
    else:
        model_name, model_params = _get_name_and_params(models)
        new_models = models
        if preprocessors is not None:
            new_preprocessors = preprocessors
            preprocessor_name, preprocessor_params = _get_name_and_params(preprocessors)

    return (
        new_models,
        model_name,
        model_params,
        new_preprocessors,
        preprocessor_name,
        preprocessor_params,
    )


def save_model_and_preprocessor(
    output_dir,
    models,
    preprocessors=None,
):
    if models is not None:
        joblib.dump(models, os.path.join(output_dir, "model.joblib"))

    if preprocessors is not None:
        if isinstance(preprocessors, list):
            preprocessors = ProcessorPipeline(preprocessors)
        joblib.dump(preprocessors, os.path.join(output_dir, "preprocessor.joblib"))


def save_feature_importances(
    output_dir,
    feat_importances,
    preprocessor,
    data,
    target,
    target_columns,
    save_plot=True,
    label_names=None,
    **kwargs,
):
    feat_importances = feat_importances.reset_index(names=["features"])
    if save_plot:
        transformed_dataset = (
            preprocessor.fit_transform(data, load_from_cache_file=False)
            if preprocessor
            else data
        )
        if kwargs.get("input_columns") is not None:
            trans_columns = set(
                DataHandler.get_column_names(transformed_dataset, generate_cols=True)
            )
            kwargs["input_columns"] = [
                col for col in kwargs["input_columns"] if col in trans_columns
            ]

        if is_bioset(data) or is_dataset(data):
            y = DataHandler.select_columns(target, columns=target_columns)
        else:
            transformed_dataset = DataHandler.to_pandas(transformed_dataset)
            y = DataHandler.to_pandas(target, columns=target_columns)

        plot_dir = os.path.join(output_dir, "plots")
        params = dict(
            y=y,
            path=plot_dir,
            target_columns=target_columns,
        )
        params.update(kwargs)

        if feat_importances.iloc[:, 1:].sum().sum() == 0:
            logger.warning(
                "All feature importances are zero. Please check the model and data. "
                "Skipping feature importance plot."
            )
        else:
            path = params.pop("path", None)
            params["output_dir"] = path
            plot_feature_importance(
                X=transformed_dataset,
                feature_importances=feat_importances,
                label_names=label_names,
                **params,
            )

    feature_metadata = None
    if "feature_metadata" in kwargs:
        feature_metadata = kwargs["feature_metadata"]
    elif is_bioset(data):
        from biosets import get_feature_metadata

        feature_metadata = get_feature_metadata(data)

    if feature_metadata is not None:
        if isinstance(feature_metadata, dict):
            feature_metadata = pd.DataFrame(
                list(feature_metadata.values()), index=list(feature_metadata.keys())
            )
            feature_metadata = feature_metadata.reset_index(names=["features"])
        if feat_importances.shape[1] > 2:
            feat_importances["median"] = feat_importances.iloc[:, 1:].median(axis=1)
        feat_importances = feature_metadata.merge(
            feat_importances, how="inner", on="features"
        )
    feat_importances.to_csv(
        os.path.join(output_dir, "feature_importances.csv"), index=False
    )


def save_study(
    output_dir,
    study,
):
    from optuna.visualization import (
        plot_param_importances,
        plot_slice,
    )

    joblib.dump(study, os.path.join(output_dir, "study.joblib"))
    trials_df = study.trials_dataframe()
    best_trial_col = [""] * len(trials_df)
    best_trial_col[study.best_trial.number] = "Best Trial"
    trials_df["Best Trial"] = best_trial_col
    trials_df.to_csv(os.path.join(output_dir, "trials.csv"), index=False)
    num_trials = len(study.trials)

    plot_dir = os.path.join(output_dir, "plots")
    param_importances_path = os.path.join(plot_dir, "param_importances_plot.pdf")
    slice_path = os.path.join(plot_dir, "slice_plot.pdf")

    os.makedirs(plot_dir, exist_ok=True)

    if is_plotly_available():
        try:
            if num_trials is not None and num_trials > 1:
                plot_param_importances(study).write_image(
                    param_importances_path, format="pdf", engine="kaleido"
                )
            plot_slice(study).write_image(slice_path, format="pdf", engine="kaleido")
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Failed to save optimization plots: {e}")
    else:
        logger.warning_once(
            "Plotly is not installed. Optimization plots will not be saved."
        )


def save_metrics(
    output_dir,
    metrics,
):
    if isinstance(metrics, list) and len(metrics) == 1:
        metrics = pd.DataFrame({k: [v] for k, v in metrics[0].items()})
    elif isinstance(metrics, dict):
        metrics = pd.DataFrame({k: [v] for k, v in metrics.items()})
    metrics.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)


def save_confusion_matrix(output_dir, confusion_matrices, label_names=None):
    if not isinstance(confusion_matrices, list) or len(confusion_matrices) == 1:
        if isinstance(confusion_matrices, list):
            confusion_matrices = confusion_matrices[0]
        confusion_matrices.to_csv(os.path.join(output_dir, "confusion_matrix.csv"))
    else:
        if label_names is None:
            label_names = [f"label_{i}" for i in range(len(confusion_matrices))]
        for i, cm in enumerate(confusion_matrices):
            cm.to_csv(
                os.path.join(output_dir, f"confusion_matrices_{label_names[i]}.csv")
            )


def save_predictions(
    output_dir,
    preds,
    data,
    target=None,
    label_names=None,
    target_columns=None,
    index=None,
):
    # save as csv
    if index is None:
        index = range(len(data))

    if label_names is None:
        if is_bioset(data) or is_dataset(data):
            if target_columns is not None:
                if isinstance(target_columns, list):
                    target_columns = target_columns[0]
                if target_columns in data._info.features:
                    label_names = data._info.features[target_columns].names
        if label_names is None and (is_bioset(target) or is_dataset(target)):
            target_columns = target_columns or DataHandler.get_column_names(target)[0]
            if target_columns is not None:
                if isinstance(target_columns, list):
                    target_columns = target_columns[0]
                if target_columns in target._info.features:
                    label_names = target._info.features[target_columns].names

    if is_bioset(data) or is_dataset(data):
        from biosets.features import Sample

        names = [
            name
            for name, feat in data._info.features.items()
            if isinstance(feat, Sample)
        ]
        if len(names) == 1:
            names = names[0]
            sample_ids = DataHandler.to_numpy(data, names).flatten()

            if sample_ids is not None:
                if index is not None and isinstance(index, (list, np.ndarray)):
                    sample_ids = np.array(sample_ids)[np.array(index)]
                if isinstance(preds.index[0], int):
                    preds.index = [sample_ids[i] for i in preds.index]
                else:
                    preds.index = sample_ids
                preds.index.name = names

    if label_names is not None:
        preds["predicted"] = DataHandler.argmax(
            preds.iloc[:, : len(label_names)], axis=1
        )
        if target is not None:
            preds["actual"] = DataHandler.to_numpy(target).flatten()[index].astype(int)
            preds["actual"] = preds["actual"].map(
                {i: name for i, name in enumerate(label_names)}
            )
            col_names = label_names + ["predicted", "actual"]

            if is_bioset(target) or is_dataset(target):
                from biosets.features import BinClassLabel

                feat = target._info.features[target_columns]
                if isinstance(feat, BinClassLabel) and (
                    feat.positive_labels is not None or feat.negative_labels is not None
                ):
                    if feat.id in DataHandler.get_column_names(data):
                        preds["actual_original"] = DataHandler.to_pandas(
                            data, feat.id
                        ).values.flatten()[index]
                        col_names.append("actual_original")
        else:
            col_names = label_names + ["predicted"]
        preds["predicted"] = preds["predicted"].map(
            {i: name for i, name in enumerate(label_names)}
        )
        preds.columns = col_names
    else:
        preds["predicted"] = DataHandler.argmax(preds, axis=1)
        if target is not None:
            preds["actual"] = (
                DataHandler.to_pandas(target).iloc[index, 0].values.astype(int)
            )

    if preds.index.name is not None:
        preds.to_csv(os.path.join(output_dir, "predictions.csv"), index=True)
    else:
        preds.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


def _save_train_results(
    data,
    orig_target,
    target,
    output_dir,
    feature_importances=None,
    confusion_matrices=None,
    study: "optuna.Study" = None,
    num_trials=None,
    time_limit=None,
    models=None,
    metrics=None,
    preds=None,
    input_columns: List[str] = None,
    target_columns: List[str] = None,
    label_names: List[str] = None,
    cv: Union[KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold] = None,
    group_name: str = None,
    outer_cv: Union[KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold] = None,
    outer_group_name: str = None,
    task: str = None,
    eval_metric: Union[str, Callable] = None,
    random_state: Union[int, List[int]] = None,
    use_suggested_hyperparameters: bool = True,
    unused_columns: List[str] = None,
    valid_split: str = None,
    index: List[int] = None,
    feature_importance_plot_params=None,
    **kwargs,
):
    orig_data = data
    data, target, _, _, _, _, _, target_columns = _get_data(
        data=data,
        target=target,
        valid_data=None,
        valid_target=None,
        groups=None,
        group_name=None,
        input_columns=input_columns,
        target_columns=target_columns,
        format=None,
        target_required=True,
    )

    if isinstance(preds, list) and len(preds) == 1:
        preds = preds[0]

    os.makedirs(output_dir, exist_ok=True)
    if study:
        save_study(output_dir, study)

    preprocessors = None
    if models is not None:
        (
            models,
            model_name,
            model_params,
            preprocessors,
            preprocessor_name,
            preprocessor_params,
        ) = _get_processor_info(models)
        save_model_and_preprocessor(output_dir, models, preprocessors)

    if metrics is not None:
        save_metrics(output_dir, metrics)

    if preds is not None:
        save_predictions(
            output_dir,
            preds,
            orig_data,
            target if orig_target is None else orig_target,
            label_names,
            target_columns,
            index,
        )

    if confusion_matrices is not None:
        save_confusion_matrix(output_dir, confusion_matrices)

    if feature_importances is not None:
        feature_importance_plot_params = feature_importance_plot_params or {}
        save_feature_importances(
            output_dir,
            feature_importances,
            preprocessors,
            orig_data,
            target,
            target_columns,
            input_columns=input_columns,
            label_names=label_names,
            **feature_importance_plot_params,
        )

    params = {
        "preprocessing": [],
        "optimization": {},
        "training": {},
        "cross_validation": {},
        "model": {},
        "misc": {},
    }

    # General Information
    if task is not None:
        params["training"]["task"] = task

    if label_names is not None:
        if not isinstance(target_columns, list):
            tc = [target_columns]
        else:
            tc = target_columns

        if not isinstance(label_names, (np.ndarray, list)) or not isinstance(
            label_names[0], (np.ndarray, list)
        ):
            label_names = [label_names]

        if "classification" in task:
            target_type = "labels"
        else:
            target_type = "targets"

        params["training"][target_type] = {}
        for i, lab_name in enumerate(tc):
            if lab_name is not None:
                class_feat = None
                if (
                    is_bioset(target) or is_dataset(target)
                ) and lab_name in target._info.features:
                    class_feat = target._info.features[lab_name]
                elif is_bioset(data) or is_dataset(data):
                    class_feat = data._info.features[lab_name]
                if class_feat is not None:
                    if "biosets" in sys.modules and isinstance(
                        class_feat, sys.modules["biosets"].BinClassLabel
                    ):
                        _names = class_feat.names
                        positive_labels = class_feat.positive_labels
                        negative_labels = class_feat.negative_labels
                        if positive_labels is not None or negative_labels is not None:
                            params["training"][target_type][lab_name] = {
                                _names[0]: negative_labels,
                                _names[1]: positive_labels,
                            }
                        else:
                            params["training"][target_type][lab_name] = _names
                    else:
                        lab_name = lab_name or f"{i + 1}"
                        params["training"][target_type][lab_name] = class_feat.names
            else:
                lab_name = lab_name or f"{i + 1}"
                params["training"][target_type][lab_name] = DataHandler.to_list(
                    label_names[i]
                )

        if len(tc) == 1:
            if tc[0] is None or tc[0] not in params["training"][target_type]:
                p = params["training"][target_type]
                if len(p):
                    params["training"][target_type] = params["training"][target_type][
                        next(iter(p))
                    ]
            else:
                params["training"][target_type] = params["training"][target_type][tc[0]]

    if target_columns is not None:
        params["training"]["target_columns"] = target_columns
    if eval_metric is not None:
        params["training"]["eval_metric"] = (
            eval_metric.__name__ if callable(eval_metric) else eval_metric
        )
    if random_state is not None:
        params["training"]["random_state"] = random_state

    if valid_split is not None:
        params["training"]["valid_split"] = valid_split

    # Hyperparameter Tuning Information
    if num_trials is not None:
        params["optimization"]["num_trials"] = num_trials
    if time_limit is not None:
        params["optimization"]["time_limit"] = time_limit
    # list tuned hyperparameters and their ranges
    if study is not None:
        search_space = study.get_trials()[0].distributions
        params["optimization"]["search_space"] = {}
        for param_name, dist in search_space.items():
            dist_pars = dist.__dict__
            dist_pars.pop("step", None)
            params["optimization"]["search_space"][param_name] = {
                "type": (dist.__class__.__name__),
                **dist_pars,
            }

    # Cross-Validation Information
    if cv is not None:
        params["cross_validation"]["cv"] = {
            "type": cv.__class__.__name__,
            "params": cv.__dict__,
        }

    if outer_cv is not None:
        params["cross_validation"]["outer_cv"] = {
            "type": outer_cv.__class__.__name__,
            "params": outer_cv.__dict__,
        }

    # Model Information
    if model_name is not None:
        params["model"]["name"] = model_name
        params["model"]["params"] = model_params if model_params is not None else {}

    # Preprocessor Information
    if preprocessor_name is not None and preprocessor_params is not None:
        if isinstance(preprocessor_name, list):
            params["preprocessing"] = [
                {"name": name, "params": params}
                for name, params in zip(preprocessor_name, preprocessor_params)
            ]
        else:
            params["preprocessing"] = {
                "name": preprocessor_name,
                "params": preprocessor_params,
            }

    if unused_columns is not None:
        params["misc"]["output_dir"] = output_dir

    if unused_columns is not None:
        params["misc"]["unused_columns"] = unused_columns

    if outer_group_name is not None:
        params["misc"]["outer_group_name"] = outer_group_name

    if group_name is not None:
        params["misc"]["group_name"] = group_name

    with open(os.path.join(output_dir, "training_params.yaml"), "w") as f:
        # uncomment for debugging
        # print(yaml.safe_dump(params, sort_keys=False))
        yaml.safe_dump(params, f, sort_keys=False)


def _init_cv(
    cv, groups=None, cv_params=None, sub_task="classification", shuffle=True, seed=42
):
    n_splits = 5
    if isinstance(cv, int):
        n_splits = cv
        if groups is not None:
            cv = LeaveOneGroupOut
        else:
            cv = next(iter(_CV[sub_task].values()))

    if isinstance(cv, str):
        cv = _CV[sub_task][cv]

    if isinstance(cv, type):
        if cv_params is None:
            if cv.__module__.split(".")[0] in ["biofit", "sklearn"]:
                cv_params = {}
                cv_params["shuffle"] = shuffle
                cv_params["random_state"] = seed if shuffle else None
                cv_params["n_splits"] = n_splits
            else:
                raise ValueError(
                    "Please provide cv_params for custom cross validation."
                )
        cv_params = get_kwargs(cv_params, cv)
        cv = cv(**cv_params)
    return cv


def get_task(models):
    if isinstance(models, list) and len(models):
        model = models[0]
    else:
        model = models
    if isinstance(model, (ProcessorPipeline, Pipeline)):
        for m in model.steps:
            val = m
            if isinstance(val, tuple):
                val = val[-1]
            if hasattr(val, "config") and hasattr(val.config, "class_names"):
                return val.config.class_names

    elif hasattr(model, "config") and hasattr(model.config, "class_names"):
        return model.config.class_names
    return None


def set_class_names(models, class_names):
    if class_names is None:
        return models
    if isinstance(models, list) and len(models):
        return [set_class_names(m, class_names) for m in models]
    if isinstance(models, (ProcessorPipeline, Pipeline)):
        for val in models.steps:
            if isinstance(val, tuple) and hasattr(val[-1], "config"):
                val[-1].config.class_names = class_names
            if hasattr(val, "config"):
                val.config.class_names = class_names
    elif hasattr(models, "config"):
        models.config.class_names = class_names
    return models


def infer_task(data=None, target=None, target_columns=None, task=None):
    sub_task = None
    if target is None and data is not None:
        _, target, _, _, _, _, _, target_columns = _get_data(
            data=data,
            target=target,
            valid_data=None,
            valid_target=None,
            groups=None,
            group_name=None,
            input_columns=None,
            target_columns=target_columns,
            format=None,
            target_required=False,
        )
    if target is None and task is None:
        raise ValueError("Target is required to infer task.")

    target_dims = DataHandler.get_shape(target) if target is not None else []
    class_names = None
    if task is None and (is_bioset(target) or is_dataset(target)):
        from biosets.features import BinClassLabel, ClassLabel, RegressionTarget

        if isinstance(target_columns, list):
            target_columns = target_columns[0]
        if target_columns and target_columns in target._info.features:
            feat = target._info.features[target_columns]
        else:
            feat = [
                feat
                for feat in target._info.features.values()
                if isinstance(feat, (BinClassLabel, ClassLabel, RegressionTarget))
            ][0]
        if isinstance(feat, (BinClassLabel, ClassLabel)):
            class_names = target._info.features[target_columns].names
            task = "classification"
        elif isinstance(feat, RegressionTarget):
            task = "regression"
    if task == "classification":
        if len(target_dims) == 1 or target_dims[1] == 1:
            n_classes = None
            if target_columns and (is_bioset(target) or is_dataset(target)):
                if isinstance(target_columns, list):
                    target_columns = target_columns[0]
                class_names = target._info.features[target_columns].names
                n_classes = len(class_names)
            else:
                n_classes = DataHandler.nunique(target)

            if n_classes > 2:
                sub_task = "multiclass_classification"
            else:
                sub_task = "binary_classification"
        else:
            sub_task = "multilabel_classification"
    elif task == "regression":
        if len(target_dims) > 1 and target_dims[1] > 1:
            sub_task = "multiregression"
        else:
            sub_task = "regression"
    elif task in CLASSIFICATION_TASKS or task in REGRESSION_TASKS:
        sub_task = task
    else:
        raise ValueError(
            "Invalid task. Please specify either a task, such as 'classification' or "
            f"'regression', or a sub task, such as {CLASSIFICATION_TASKS + REGRESSION_TASKS}."
        )
    return sub_task, class_names


def _iter_preprocessor(preprocessor, condition=None):
    if isinstance(preprocessor, (Pipeline, ProcessorPipeline)):
        start = False
        for n, p in preprocessor.steps:
            if isinstance(p, BaseProcessor):
                if condition == "sample independent":
                    if p.has_fit:
                        yield p.config.processor_name, p
                    else:
                        break
                elif condition == "sample dependent":
                    if not p.has_fit:
                        start = True
                        yield p
                    elif start:
                        yield p
                else:
                    yield p
            else:
                yield p
    elif isinstance(preprocessor, list):
        for p in preprocessor:
            yield p
    else:
        yield preprocessor


def preprocess(
    preprocessor,
    x_train,
    y_train=None,
    x_valid=None,
    y_valid=None,
    cache_dir=None,
    transform_only=False,
    raise_error=False,
):
    try:
        if isinstance(preprocessor, (Pipeline, ProcessorPipeline)):
            for proc in preprocessor.steps:
                p = proc[-1] if isinstance(proc, tuple) else proc
                if isinstance(p, BaseProcessor):
                    extra_kwargs = {
                        "cache_dir": cache_dir,
                        "load_from_cache_file": False,
                    }
                else:
                    extra_kwargs = {}
                if transform_only:
                    x_train = p.transform(x_train, **extra_kwargs)
                else:
                    x_train = p.fit_transform(x_train, **extra_kwargs)
                if x_valid is not None:
                    x_valid = p.transform(x_valid, **extra_kwargs)
        else:
            if isinstance(preprocessor, BaseProcessor):
                extra_kwargs = {"cache_dir": cache_dir, "load_from_cache_file": False}
            else:
                extra_kwargs = {}
            if transform_only:
                x_train = preprocessor.transform(x_train, **extra_kwargs)
            else:
                x_train = preprocessor.fit_transform(x_train, **extra_kwargs)
            if x_valid is not None:
                x_valid = preprocessor.transform(x_valid, **extra_kwargs)
    except ValueError:
        if raise_error:
            raise
        else:
            logger.info("Preprocessing failed, using nan_to_num")
            _train_cols = x_train.columns.tolist()
            x_train = np.nan_to_num(x_train)
            # convert back to dataframe
            x_train = pd.DataFrame(x_train, columns=_train_cols)
            if x_valid is not None:
                _valid_cols = x_valid.columns.tolist()
                x_valid = np.nan_to_num(x_valid)
                x_valid = pd.DataFrame(x_valid, columns=_valid_cols)
            return preprocess(
                preprocessor,
                x_train,
                y_train,
                x_valid,
                y_valid,
                cache_dir,
                transform_only,
                True,
            )

    return x_train, y_train, x_valid, y_valid


def split(cv, X, y=None, groups=None, indices=None):
    if cv is None:
        return [(None, None)]
    # check if cv is a generator
    if isinstance(cv, Generator):
        return cv
    if indices is not None:
        return cv.split(
            DataHandler.select_rows(X, indices),
            y=DataHandler.select_column(DataHandler.select_rows(y, indices), 0)
            if y is not None
            else None,
            groups=DataHandler.select_column(
                DataHandler.select_rows(groups, indices), 0
            )
            if groups is not None
            else None,
        )
    return cv.split(
        X,
        y=DataHandler.select_column(y, 0) if y is not None else None,
        groups=DataHandler.select_column(groups, 0) if groups is not None else None,
    )


def get_model(models_or_name, task=None):
    if isinstance(models_or_name, list) and len(models_or_name):
        mon = models_or_name[0]
    else:
        mon = models_or_name

    if mon is None:
        return None

    model = None
    model_cls = None
    if isinstance(mon, str):
        model_name = mon
        model_cls = _MODELS[task][model_name]
    elif isinstance(mon, type):
        model_cls = mon
        model_name = mon.__module__.split(".")[-1]
    elif isinstance(mon, Model):
        model_name = mon.config.processor_name
        model_cls = model.__class__
        model = mon
    elif isinstance(mon, (ProcessorPipeline, Pipeline)):
        for m in _flatten_pipeline(mon):
            out = get_model(m, task)
            if out is not None:
                return out
        return None
    elif hasattr(mon, "predict"):
        model_cls = mon.__class__
        model_name = mon.__class__.__module__.split(".")[-1]
        model = mon
    else:
        return None
    return model, model_cls, model_name


def get_model_info(model, task=None):
    out = get_model(model, task=None)
    if out is not None:
        model, _, _ = out
        if hasattr(model, "config"):
            return asdict(model.config)
        elif hasattr(model, "get_params"):
            return model.get_params()
        else:
            return model.__dict__

    return {}


def _split_data(
    data, target, valid_split, seed, shuffle, stratify_by_column, keep_in_memory=True
):
    def _convert_to_in_memory(data, indices=None):
        from datasets import InMemoryTable, MemoryMappedTable

        try:
            if data._indices or indices is not None:
                if indices is None:
                    indices = data._indices.column("indices")

                if isinstance(data._data, MemoryMappedTable):
                    table = MemoryMappedTable._apply_replays(
                        data._data.table, data._data.replays
                    )
                else:
                    table = data._data.table
                table = table.take(indices)
                data._indices = None
                data._data = InMemoryTable(table)

                return data
        except pa.ArrowInvalid:
            logger.error(
                "Table is too large for row selection. Please set keep_in_memory=False or "
                "shuffle=False until https://github.com/apache/arrow/issues/25822 has been resolved"
            )
            raise

    if is_bioset(data) or is_dataset(data):
        if keep_in_memory:
            _convert_to_in_memory(data)

        train_data, valid_data = data.train_test_split(
            test_size=valid_split,
            seed=seed if shuffle else None,
            shuffle=shuffle,
            stratify_by_column=stratify_by_column if shuffle else None,
        ).values()
        if not shuffle and not stratify_by_column:
            target, valid_target = target.train_test_split(
                test_size=valid_split,
                seed=seed,
                shuffle=shuffle,
                stratify_by_column=stratify_by_column,
            )
        else:
            train_indices = train_data._indices.column("indices")
            if valid_data._indices:
                valid_indices = valid_data._indices.column("indices")
            else:
                valid_indices = pa.array(
                    list(
                        sorted(
                            set(range(train_data.num_rows))
                            - set(DataHandler.to_numpy(train_indices).tolist())
                        )
                    )
                )

            if keep_in_memory:
                train_data = _convert_to_in_memory(train_data, train_indices)
                valid_data = _convert_to_in_memory(valid_data, valid_indices)
                target = _convert_to_in_memory(target, train_indices)
                valid_target = _convert_to_in_memory(valid_target, valid_indices)
            else:
                valid_target = copy.deepcopy(target)
                valid_target = valid_target.select(valid_indices)
                target = target.select(train_indices)
    else:
        if stratify_by_column is not None:
            stratify = DataHandler.to_numpy(data, stratify_by_column)
        else:
            stratify = target
        train_data, valid_data, target, valid_target = train_test_split(
            data,
            target,
            test_size=valid_split,
            random_state=seed,
            shuffle=shuffle,
            stratify=stratify,
        )
    return train_data, valid_data, target, valid_target


def _get_label_names(data, target, target_columns, u_labels=None):
    if isinstance(target_columns, list):
        tc = target_columns[0]
    else:
        tc = target_columns
    if (is_bioset(data) or is_dataset(data)) and tc in data._info.features:
        labels = data._info.features[tc].names
    elif (is_bioset(target) or is_dataset(target)) and tc in target._info.features:
        labels = target._info.features[tc].names
    else:
        labels = DataHandler.to_list(u_labels)
    return labels


def _get_data(
    data,
    target=None,
    valid_data=None,
    valid_target=None,
    valid_split=None,
    groups=None,
    outer_groups=None,
    group_name=None,
    outer_group_name=None,
    input_columns="auto",
    target_columns="auto",
    valid_input_columns="auto",
    valid_target_columns="auto",
    target_required=True,
    shuffle=True,
    seed=42,
    keep_in_memory=True,
    format="pandas",
):
    if DataHandler.supports_named_columns(data):
        use_auto_target_columns = target_columns == "auto"
    else:
        input_columns = None if input_columns == "auto" else input_columns
        target_columns = None if target_columns == "auto" else target_columns

    if valid_data is not None and not DataHandler.supports_named_columns(valid_data):
        valid_input_columns = (
            None if valid_input_columns == "auto" else valid_input_columns
        )
        valid_target_columns = (
            None if valid_target_columns == "auto" else valid_target_columns
        )

    def get_target_if_none(data, target_columns):
        target = None
        if target_columns is not None:
            sel_kwargs = {}
            if is_bioset(data) or is_dataset(data):
                sel_kwargs = {"keep_old_fingerprint": False}
            if isinstance(target_columns, str):
                target = DataHandler.select_columns(
                    data, [target_columns], **sel_kwargs
                )
            else:
                target = DataHandler.select_columns(data, target_columns, **sel_kwargs)
        return target

    def handle_target_columns(data, target_columns, required=True):
        if target_columns == "auto":
            if is_bioset(data):
                from biosets import get_target_col_names

                target_columns = get_target_col_names(data)
            elif target_required:
                ValueError(
                    "`data` must be a `Dataset` to automatically infer target columns. "
                    "Please provide `target_columns` or `target`."
                )
        if isinstance(target_columns, (type, tuple)):
            if is_bioset(data) or is_dataset(data):
                target_columns = [
                    k
                    for k, v in data._info.features.items()
                    if isinstance(v, target_columns)
                ]
        if target_columns == "auto":
            target_columns = None
        if target_columns is None and target_required:
            raise ValueError("Target columns must be provided if target is None.")

        if isinstance(target_columns, list) and len(target_columns) == 1:
            target_columns = target_columns[0]
        return target_columns

    def handle_input_columns(data, input_columns):
        if is_bioset(data) or is_dataset(data):
            if input_columns == "auto" and is_bioset(data):
                from biosets import get_data_col_names

                input_columns = get_data_col_names(data)
            elif isinstance(input_columns, (type, tuple)):
                input_columns = [
                    k
                    for k, v in data._info.features.items()
                    if isinstance(v, input_columns)
                ]
        if input_columns == "auto":
            input_columns = None

        return input_columns

    x_train = None
    y_train = None
    y_valid = None
    x_valid = None

    if isinstance(data, dict):
        if isinstance(valid_split, str):
            valid_data = data.get(valid_split)
        else:
            valid_data = data.get(
                "valid", data.get("test"), data.get("validation"), None
            )
        if len(data) < 3:
            train_split = [
                k
                for k in data.keys()
                if k not in ["valid", "test", "validation", valid_split]
            ]
            if not train_split:
                raise ValueError("Train split not found in dataset.")
            data = data.get(train_split)
        else:
            data = data.get("train", data.get("training"))

    if target is None:
        target_columns = handle_target_columns(
            data, target_columns, required=target_required
        )
        y_train = get_target_if_none(data, target_columns)

    else:
        y_train = target
        if DataHandler.supports_named_columns(target):
            target_columns = DataHandler.get_column_names(target)[0]

    if valid_target is None and valid_data is not None:
        if valid_target_columns is None or valid_target_columns == "auto":
            valid_target_columns = handle_target_columns(
                valid_data,
                "auto" if use_auto_target_columns else target_columns,
                required=False,
            )
        if valid_target_columns is not None:
            y_valid = get_target_if_none(valid_data, valid_target_columns)
        else:
            y_valid = get_target_if_none(valid_data, target_columns)
        if valid_target_columns is not None and target_columns is not None:
            if valid_target_columns != target_columns:
                valid_target = DataHandler.set_column_names(
                    valid_target, target_columns
                )
        elif valid_target_columns is not None:
            target_columns = valid_target_columns

    elif valid_target is not None:
        y_valid = valid_target

    if group_name is not None and groups is None:
        groups = DataHandler.select_column(data, group_name)
    if outer_group_name is not None and outer_groups is None:
        outer_groups = DataHandler.select_column(data, outer_group_name)
    if (
        group_name is None
        and groups is not None
        and DataHandler.supports_named_columns(groups)
    ):
        group_name = DataHandler.get_column_names(groups)[0]
    if (
        outer_group_name is None
        and outer_groups is not None
        and DataHandler.supports_named_columns(outer_groups)
    ):
        outer_group_name = DataHandler.get_column_names(outer_groups)[0]

    if valid_data is None and valid_split is not None:
        data, valid_data, y_train, y_valid = _split_data(
            data,
            y_train,
            valid_split,
            seed=seed,
            shuffle=shuffle,
            stratify_by_column=group_name,
            keep_in_memory=keep_in_memory,
        )

    input_columns = handle_input_columns(data, input_columns)
    if valid_data is not None:
        valid_input_columns = handle_input_columns(valid_data, valid_input_columns)
    if input_columns is not None:
        sel_kwargs = {}
        if is_bioset(data) or is_dataset(data):
            sel_kwargs = {"keep_old_fingerprint": False}
        x_train = DataHandler.select_columns(data, input_columns, **sel_kwargs)
        if valid_data is not None:
            sel_kwargs = {}
            if is_bioset(valid_data) or is_dataset(valid_data):
                sel_kwargs = {"keep_old_fingerprint": False}
            valid_input_columns = valid_input_columns or input_columns
            x_valid = DataHandler.select_columns(
                valid_data, valid_input_columns, **sel_kwargs
            )
    elif (
        target is not None
        and DataHandler.supports_named_columns(data)
        and DataHandler.supports_named_columns(target)
    ):
        _input_columns = DataHandler.get_column_names(data)
        _target_columns = DataHandler.get_column_names(target)
        if group_name is not None:
            _target_columns.append(group_name)
        if outer_group_name is not None:
            _target_columns.append(outer_group_name)
        _intersecting_cols = set(_input_columns) & set(_target_columns)
        if _intersecting_cols:
            x_train = DataHandler.drop_columns(data, list(_intersecting_cols))
            if (
                valid_data is not None
                and DataHandler.supports_named_columns(valid_data)
                and DataHandler.supports_named_columns(x_train)
            ):
                sel_kwargs = {}
                if is_bioset(valid_data) or is_dataset(valid_data):
                    sel_kwargs = {"keep_old_fingerprint": False}
                x_valid = DataHandler.select_columns(
                    valid_data, DataHandler.get_column_names(x_train), **sel_kwargs
                )
        else:
            x_train = data
            if valid_data is not None:
                x_valid = valid_data
        input_columns = DataHandler.get_column_names(x_train)
    else:
        x_train = data
        if valid_data is not None:
            x_valid = valid_data
        input_columns = DataHandler.get_column_names(x_train)

    if format is not None:
        x_train = DataHandler.to_format(x_train, format)
        if y_train is not None:
            y_train = DataHandler.to_format(y_train, format)
        if groups is not None:
            groups = DataHandler.to_format(groups, format)
        if outer_groups is not None:
            outer_groups = DataHandler.to_format(outer_groups, format)
        if x_valid is not None:
            x_valid = DataHandler.to_format(x_valid, format)
        if y_valid is not None:
            y_valid = DataHandler.to_format(y_valid, format)

    return (
        x_train,
        y_train,
        x_valid,
        y_valid,
        groups,
        outer_groups,
        input_columns,
        target_columns,
    )
