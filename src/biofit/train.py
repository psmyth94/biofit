import copy
import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Union

import numpy as np
import pandas as pd
from biocore import DataHandler
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    BaseCrossValidator,
)
from sklearn.pipeline import Pipeline

from biofit.auto.processing_auto import ProcessorPipeline
from biofit.processing import BaseProcessor
from biofit.train_eval_utils import (
    _flatten_pipeline,
    _get_data,
    split,
)
from biofit.utils import (
    enable_full_determinism,
    logging,
)

if TYPE_CHECKING:
    import polars as pl
    from datasets import Dataset

logger = logging.get_logger(__name__)


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
                        # "load_from_cache_file": False,
                    }
                    # p.config.enable_caching = False
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
                extra_kwargs = {
                    "cache_dir": cache_dir,
                    # "load_from_cache_file": False,
                }
                # p.config.enable_caching = False
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


def train(
    model: Union[BaseEstimator, BaseProcessor],
    data: Union[pd.DataFrame, "pl.DataFrame", "Dataset"],
    target: Union[
        pd.Series, "pl.Series", pd.DataFrame, "pl.DataFrame", "Dataset"
    ] = None,
    valid_data: Union[pd.DataFrame, "pl.DataFrame", "Dataset"] = None,
    valid_target: Union[
        pd.Series, "pl.Series", pd.DataFrame, "pl.DataFrame", "Dataset"
    ] = None,
    groups: Union[pd.Series, "pl.Series"] = None,
    input_columns: Union[List[str], str] = "auto",
    target_columns: Union[List[str], str] = "auto",
    group_name: str = None,
    preprocessor: Union[BaseEstimator, BaseProcessor] = None,
    eval_metric: Union[str, Callable] = None,
    task: str = None,
    cv: BaseCrossValidator = None,
    random_state: Union[List[int], int] = 42,
    save_indices: bool = False,
    output_dir: str = None,
    cache_dir: str = None,
):
    """
    Train a model or processor on the provided data, with optional preprocessing,
    validation data, and cross-validation.

    This function supports training models with various data formats (pandas DataFrame,
    polars DataFrame, Dataset), optional preprocessing pipelines, cross-validation
    strategies, and random seeds for reproducibility.

    Parameters:
        model (Union[BaseEstimator, BaseProcessor]):
            The model or processor to train. This can be a scikit-learn estimator, a
            pipeline, or a custom estimator that follows scikit-learn's API.
        data (Union[pd.DataFrame, pl.DataFrame, "Dataset"]):
            The training data. It can be a pandas DataFrame, polars DataFrame, or a
            Dataset object.
        target (Union[pd.Series, pl.Series, pd.DataFrame, pl.DataFrame, "Dataset"], optional):
            The target values for supervised learning. If None, the target columns must
            be specified in `data` using `target_columns`.
        valid_data (Union[pd.DataFrame, pl.DataFrame, "Dataset"], optional):
            Optional validation data. If not provided, validation can be done using
            cross-validation.
        valid_target (Union[pd.Series, pl.Series, pd.DataFrame, pl.DataFrame, "Dataset"], optional):
            Target values for the validation data.
        groups (Union[pd.Series, pl.Series], optional):
            Group labels for the samples used while splitting the dataset into
            train/test set.
        input_columns (Union[List[str], str], optional):
            Names of the input feature columns. If 'auto', the function will attempt to
            infer the input columns from the data. Defaults to "auto".
        target_columns (Union[List[str], str], optional):
            Names of the target columns in `data`. If 'auto', the function will attempt
            to infer the target columns. Defaults to "auto".
        group_name (str, optional):
            Name of the group column in `data` if groups are specified within the data.
        preprocessor (Union[BaseEstimator, BaseProcessor], optional):
            Preprocessing pipeline or estimator to apply before training the model.
        eval_metric (Union[str, Callable], optional):
            Evaluation metric or a callable function to evaluate the model's
            performance.
        task (str, optional):
            Type of task to perform (e.g., 'classification', 'regression',
            'multilabel_classification', 'multi_regression').
        cv (type of BaseCrossValidator, optional):
            Cross-validation splitting strategy. If provided, the model will be trained
            using cross-validation.
        random_state (Union[List[int], int], optional):
            Random seed(s) for reproducibility. Can be a single integer or a list of
            integers. Defaults to 42.
        save_indices (bool, optional):
            Whether to save the indices of the train and validation splits. Defaults to
            False.
        output_dir (str, optional):
            Directory where outputs will be saved. Defaults to None.
        cache_dir (str, optional):
            Directory where cache files will be stored. Defaults to None.

    Returns:
        Trained model or pipeline. The return type depends on the inputs:

        - If `random_state` is an integer and `cv` is None, returns a single trained model or pipeline.
        - If `random_state` is a list of integers, returns a list of trained models or pipelines, one for each random seed.
        - If `cv` is specified, returns a list of trained models or pipelines, one for each cross-validation fold.

    Examples:
        1. Basic usage with pandas DataFrame:

        ```python
        from biofit.trainer import train
        from biofit.models import LogisticRegressionForClassification
        import pandas as pd

        # Create sample data
        data = pd.DataFrame({
            'age': [25, 32, 47, 51, 62],
            'salary': [50000, 60000, 70000, 80000, 90000],
            'purchased': [0, 1, 0, 1, 0]
        })

        # Train the model
        model = train(
            model=LogisticRegressionForClassification(),
            data=data,
            target_columns='purchased',
            input_columns=['age', 'salary']
        )
        ```

        2. Using validation data:

        ```python
        from biofit.train import train
        from biofit.models import RandomForestForClassification
        import pandas as pd

        # Training data
        train_data = pd.DataFrame({
            'feature': [1,2,3,4,5],
            'target': [2,4,6,8,10]
        })

        # Validation data
        valid_data = pd.DataFrame({
            'feature': [6,7],
            'target': [12,14]
        })

        # Train the model with validation data
        model = train(
            model=RandomForestForClassification(),
            data=train_data,
            valid_data=valid_data,
            target_columns='target',
            input_columns='feature'
        )
        ```

        3. Cross-validation with random seed list:

        ```python
        from biofit.train import train
        from biofit.models import RandomForestForClassifier
        from sklearn.model_selection import StratifiedKFold
        import pandas as pd

        # Sample data
        data = pd.DataFrame({
            'feature1': [1,2,3,4,5,6],
            'feature2': [6,5,4,3,2,1],
            'target': [0,1,0,1,0,1]
        })

        cv = StratifiedKFold(n_splits=3)
        random_seeds = [42, 7, 21]

        # Train the model with cross-validation and multiple random seeds
        models = train(
            model=RandomForestForClassifier(),
            data=data,
            target_columns='target',
            input_columns=['feature1', 'feature2'],
            cv=cv,
            random_state=random_seeds
        )

        # models is a list of models trained with different random seeds
        ```

        4. Using a preprocessor pipeline:

        ```python
        from biofit.train import train
        from biofit.auto.processing_auto import ProcessorPipeline
        from biofit.models import LogisticRegressionForClassification
        from sklearn.compose import ColumnTransformer
        import pandas as pd

        # Sample data with categorical feature
        data = pd.DataFrame({
            'age': [25, 32, 47, 51, 62],
            'gender': ['M', 'F', 'M', 'F', 'M'],
            'purchased': [0, 1, 0, 1, 0]
        })

        # Define preprocessing
        numeric_features = ['age']
        numeric_transformer = StandardScaler()

        categorical_features = ['gender']
        categorical_transformer = OneHotEncoder()

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Create a pipeline that first preprocesses the data, then applies the model
        pipeline = ProcessorPipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegressionForClassification())
        ])

        # Train the model with preprocessor
        model = train(
            model=pipeline,
            data=data,
            target_columns='purchased',
            input_columns=['age', 'gender']
        )
        ```

        5. Training with a Dataset object:

        ```python
        from biofit.train import train
        from datasets import load_dataset

        # Load a dataset from the datasets library
        dataset = load_dataset('csv', data_files='data.csv')

        # Train the model
        model = train(
            model=SomeModel(),
            data=dataset['train'],
            target_columns='label',
            input_columns='text'
        )
        ```

    Notes:
        - The `train` function can handle various data formats and will attempt to
            process the data accordingly.
        - If `input_columns` or `target_columns` are set to 'auto', the function will
            try to infer the columns automatically based on the data.
        - The `preprocessor` can be any transformer that follows the scikit-learn API,
            including pipelines.
        - When using cross-validation (`cv` is not None), the function will train models
            for each fold and return a list of models.
        - If multiple random seeds are provided (`random_state` is a list), the function
            will train multiple models for feature importance analysis, but only return
            the models trained with the first random seed.
    """
    x_train, y_train, x_valid, y_valid, groups, _, input_columns, target_columns = (
        _get_data(
            data=data,
            target=target,
            valid_data=valid_data,
            valid_target=valid_target,
            groups=groups,
            group_name=group_name,
            input_columns=input_columns,
            target_columns=target_columns,
            format="pandas",
            target_required=True,
        )
    )
    if cache_dir is None and output_dir is not None:
        cache_dir = (Path(output_dir) / ".cache").resolve().as_posix()

    def fit(
        x_train,
        y_train,
        x_valid,
        y_valid,
        model,
        preprocessor,
        random_state=None,
        cache_dir=None,
    ):
        if isinstance(model, Pipeline):
            if preprocessor is None and len(model.steps) > 1:
                preprocessor = _flatten_pipeline(
                    [p[-1] if isinstance(p, tuple) else p for p in model.steps[:-1]]
                )
                preprocessor = Pipeline(preprocessor)
            model = (
                model.steps[-1][1]
                if isinstance(model.steps[-1], tuple)
                else model.steps[-1]
            )
        if preprocessor:
            x_train, y_train, x_valid, y_valid = preprocess(
                preprocessor, x_train, y_train, x_valid, y_valid, cache_dir
            )

        m_params = (
            model.get_params() if hasattr(model, "get_params") else model.__dict__
        )
        if hasattr(model, "set_params"):
            if "random_state" in m_params:
                model.set_params(random_state=random_state)
            elif "seed" in m_params:
                model.set_params(seed=random_state)
        else:
            if "random_state" in m_params:
                model.random_state = random_state
            elif "seed" in m_params:
                model.seed = random_state
        models = []
        if task in ("multilabel_classification", "multi_regression"):
            # also multi_column_regression
            models = [model] * y_train.shape[1]
            for idx, _m in enumerate(models):
                if _m.__module__.startswith("xgboost"):
                    _m.fit(
                        x_train,
                        y_train[:, idx],
                        model__eval_set=[(x_valid, y_valid[:, idx])]
                        if x_valid is not None
                        else None,
                        model__verbose=False,
                    )
                elif _m.__module__.startswith("biofit"):
                    old_val = _m.config.enable_caching
                    _m.config.enable_caching = False
                    if (
                        "early_stopping_rounds" in m_params
                        and m_params["early_stopping_rounds"] is not None
                        and m_params["early_stopping_rounds"] > 0
                    ):
                        _m.fit(
                            x_train,
                            y_train[:, idx],
                            eval_set=[(x_valid, y_valid[:, idx])]
                            if x_valid is not None
                            else None,
                            # load_from_cache_file=False,
                        )
                    else:
                        _m.fit(
                            x_train,
                            y_train[:, idx],
                            load_from_cache_file=False,
                        )
                    _m.config.enable_caching = old_val
                else:
                    _m.fit(
                        x_train,
                        y_train[:, idx],
                    )

        else:
            if model.__module__.startswith("xgboost"):
                model.fit(
                    x_train,
                    y_train,
                    model__eval_set=[(x_valid, y_valid)],
                    model__verbose=False,
                )
            elif model.__module__.startswith("biofit"):
                old_val = model.config.enable_caching
                model.config.enable_caching = False
                model.fit(
                    x_train,
                    y_train,
                    load_from_cache_file=False,
                )
                model.config.enable_caching = old_val
            else:
                model.fit(x_train, y_train)

            models = [model]
        return models, preprocessor

    count = 0
    count += 1

    def training_loop(random_state):
        enable_full_determinism(random_state)
        if cv is None:
            _m = copy.deepcopy(model)
            _p = copy.deepcopy(preprocessor)

            fitted_model_, preprocessor_ = fit(
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                model=_m,
                preprocessor=_p,
                random_state=random_state,
                cache_dir=cache_dir,
            )

            if isinstance(fitted_model_, list) and len(fitted_model_) == 1:
                fitted_model_ = fitted_model_[0]
            return fitted_model_, preprocessor_
        else:
            fitted_models = []
            fitted_preprocessors = []
            if hasattr(cv, "__dict__") and "random_state" in cv.__dict__:
                cv.random_state = random_state
            for fold, (train_idx, valid_idx) in enumerate(
                split(cv, X=x_train, y=y_train, groups=groups)
            ):
                _cache_dir = os.path.join(cache_dir, f"fold_{fold}")
                _m = copy.deepcopy(model)
                _p = copy.deepcopy(preprocessor)

                m_params = _m.get_params()
                if "random_state" in m_params:
                    _m.set_params(random_state=random_state)
                elif "seed" in m_params:
                    _m.set_params(seed=random_state)
                xtrain_fold, xvalid_fold = (
                    DataHandler.select_rows(x_train, train_idx),
                    DataHandler.select_rows(x_train, valid_idx),
                )
                ytrain_fold, yvalid_fold = (
                    DataHandler.select_rows(y_train, train_idx),
                    DataHandler.select_rows(y_train, valid_idx),
                )
                fitted_model_fold, fitted_preprocessor_fold = fit(
                    xtrain_fold,
                    ytrain_fold,
                    xvalid_fold,
                    yvalid_fold,
                    _m,
                    _p,
                    random_state=random_state,
                    cache_dir=_cache_dir,
                )
                if isinstance(fitted_model_fold, list) and len(fitted_model_fold) == 1:
                    fitted_model_fold = fitted_model_fold[0]
                fitted_models.append(fitted_model_fold)
                fitted_preprocessors.append(fitted_preprocessor_fold)

        return fitted_models, fitted_preprocessors

    def create_pipeline(model, preprocessor):
        pipeline = []
        if isinstance(model, list):
            for m, p in zip(model, preprocessor):
                if isinstance(p, (Pipeline, ProcessorPipeline)):
                    p = ProcessorPipeline(_flatten_pipeline(p))
                pipeline.append(
                    Pipeline([("preprocessor", p), ("model", m)]) if p else m
                )
        else:
            if isinstance(preprocessor, (Pipeline, ProcessorPipeline)):
                preprocessor = ProcessorPipeline(_flatten_pipeline(preprocessor))
            pipeline = (
                Pipeline([("preprocessor", preprocessor), ("model", model)])
                if preprocessor
                else model
            )
        return pipeline

    if isinstance(random_state, list):
        pipeline_list = []
        for rs in random_state:
            m, p = training_loop(rs)
            pipeline_list.append(create_pipeline(m, p))

        return pipeline_list
    else:
        fitted_model, fitted_preprocessor = training_loop(random_state)
        return create_pipeline(fitted_model, fitted_preprocessor)
