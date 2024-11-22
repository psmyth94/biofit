import copy
import inspect
import shutil
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Type, Union
from unittest.mock import patch

import biofit.config
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from biocore import DataHandler
from biofit.integration.biosets import get_feature
from biofit.processing import (
    BaseProcessor,
    NonExistentCacheError,
    ProcessorConfig,
    SelectedColumnTypes,
    sync_backup_config,
)
from biofit.utils import version
from biofit.utils.fingerprint import generate_cache_dir

from tests.utils import create_bioset, require_biosets, require_datasets, require_polars

# Mock feature types for testing purposes
FEATURE_TYPES = Union[Type, tuple]

pytestmark = pytest.mark.unit


@dataclass
class MockProcessorConfig(ProcessorConfig):
    """
    Configuration for the MockModel. Inherits from ProcessorConfig and specifies
    feature types for automatic column selection.
    """

    # Specifies the input feature types for the fit method (arity of 2: X and y)
    _fit_input_feature_types: List[FEATURE_TYPES] = field(
        default_factory=lambda: [None, get_feature("TARGET_FEATURE_TYPES")],
        init=False,
        repr=False,
    )
    # Specifies the unused feature types during fitting
    _fit_unused_feature_types: List[FEATURE_TYPES] = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES"), None],
        init=False,
        repr=False,
    )
    # Specifies the unused feature types during transformation
    _transform_unused_feature_types: List[FEATURE_TYPES] = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES")],
        init=False,
        repr=False,
    )
    _fit_process_desc: str = field(default="Fitting test", init=False, repr=False)
    _predict_process_desc: str = field(default="Predict test", init=False, repr=False)
    _transform_process_desc: str = field(
        default="Transforming test", init=False, repr=False
    )
    # Name of the processor
    processor_type: str = field(default="mock", init=False, repr=False)
    processor_name: str = field(default="processor", init=False, repr=False)
    # Additional parameters for testing
    processor_param_int: int = None
    processor_param_float: float = None
    processor_param_str: str = None
    processor_param_bool: bool = None
    processor_param_list: List = None
    processor_param_dict: dict = None
    processor_param_tuple: tuple = None


class MockModel(BaseProcessor):
    """
    A mock model class for testing ProcessorConfig, TransformationMixin, and BaseProcessor.
    Implements the necessary _fit_* and _predict_* methods.
    """

    _config_class = MockProcessorConfig
    config: MockProcessorConfig

    def __init__(self, config: Optional[MockProcessorConfig] = None, **kwargs):
        # Initialize the BaseProcessor with the given configuration
        super().__init__(config=config or self._config_class(**kwargs))
        self.post_fit = lambda x: x

    @sync_backup_config
    def set_params(self, **kwargs):
        self.config = self.config.replace_defaults(**kwargs)
        self.post_fit = lambda x: x

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
    ) -> "MockModel":
        """
        Mock fit method that processes input data and fits the model.
        """
        # Prepare input columns (arity of 2: X and y)
        self.config._input_columns = self._set_input_columns_and_arity(
            input_columns, target_column
        )
        return self._process_fit(
            X,
            y,
            raise_if_missing=raise_if_missing,
            cache_output=cache_output,
            cache_dir=cache_dir,
            cache_file_name=cache_file_name,
            load_from_cache_file=load_from_cache_file,
            batched=batched,
            batch_size=batch_size,
            batch_format=batch_format,
            map_kwargs=map_kwargs,
            num_proc=num_proc,
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
        """
        Mock predict method that generates predictions.
        """
        self._method_prefix = "_predict"
        self.config._n_features_out = 1
        self._input_columns = self._set_input_columns_and_arity(input_columns)
        self.output_dtype = "float64"
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
        """
        Mock predict_proba method that generates predictions.
        """
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

    def _fit_sklearn(self, X, y, some_param=None):
        """
        Internal fit method for sklearn-compatible data formats.
        """
        # Mock fitting logic (e.g., training a simple model)
        self.config.processor_param_numpy = np.array([1, 2, 3])
        self.config.processor_param_arrow = pa.table({"a": [1, 2, 3]})
        self.config.processor_param_polars = pa.table({"a": [1, 2, 3]})
        self.config.processor_param_pandas = pa.table({"a": [1, 2, 3]})
        self.config.estimator = {"mean": np.mean(DataHandler.to_numpy(y).flatten())}
        return self

    def _predict_sklearn(self, X):
        """
        Internal predict method for sklearn-compatible data formats.
        """
        if not self.config.is_fitted:
            raise ValueError("Model is not fitted yet.")
        # Mock prediction logic (e.g., returning the mean value)
        predictions = np.full(len(X), self.config.estimator["mean"])
        return predictions

    def _predict_proba_sklearn(self, X):
        """
        Internal predict_proba method for sklearn-compatible data formats.
        """
        if not self.config.is_fitted:
            raise ValueError("Model is not fitted yet.")
        # Mock prediction logic (e.g., returning the mean value)
        predictions = np.full(
            (len(X), self.config.n_classes), self.config.estimator["mean"]
        )
        return predictions


class MockPreprocessor(BaseProcessor):
    """
    A mock model class for testing ProcessorConfig, TransformationMixin, and BaseProcessor.
    Implements the necessary _fit_* and _predict_* methods.
    """

    _config_class = MockProcessorConfig
    config: MockProcessorConfig

    def __init__(self, config: Optional[MockProcessorConfig] = None, **kwargs):
        # Initialize the BaseProcessor with the given configuration
        super().__init__(config=config or self.config_class(**kwargs))
        self.post_fit = lambda x: x

    @sync_backup_config
    def set_params(self, **kwargs):
        self.config = self.config.replace_defaults(**kwargs)
        self.post_fit = lambda x: x

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
    ) -> "MockModel":
        """
        Mock fit method that processes input data and fits the model.
        """
        # Prepare input columns (arity of 2: X and y)
        self.config._input_columns = self._set_input_columns_and_arity(
            input_columns, target_column
        )
        return self._process_fit(
            X,
            y,
            raise_if_missing=raise_if_missing,
            cache_output=cache_output,
            cache_dir=cache_dir,
            cache_file_name=cache_file_name,
            load_from_cache_file=load_from_cache_file,
            batched=batched,
            batch_size=batch_size,
            batch_format=batch_format,
            map_kwargs=map_kwargs,
            num_proc=num_proc,
            fingerprint=fingerprint,
        )

    def transform(
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
        """
        Mock predict method that generates predictions.
        """
        self._method_prefix = "_transform"
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

    def fit_transform(
        self,
        X,
        y=None,
        input_columns: SelectedColumnTypes = None,
        target_column: SelectedColumnTypes = None,
        keep_unused_columns: bool = None,
        raise_if_missing: bool = None,
        cache_output: bool = None,
        load_from_cache_file: bool = None,
        batched: bool = True,
        batch_size: int = 1000,
        output_format: str = None,
        batch_format: str = None,
        num_proc: int = None,
        map_kwargs: dict = {"fn_kwargs": {}},
        cache_dir: str = None,
        cache_file_name: str = None,
        fingerprint: str = None,
    ):
        """
        Mock fit_transform method that processes input data and fits the model.
        """
        return self.fit(
            X,
            y,
            input_columns=input_columns,
            target_column=target_column,
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
        ).transform(
            X,
            input_columns=input_columns,
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

    def _fit_pandas(self, X, y):
        """
        Internal fit method for sklearn-compatible data formats.
        """
        return self

    def _fit_arrow(self, X, y):
        """
        Internal fit method for sklearn-compatible data formats.
        """
        return self

    def _fit_polars(self, X, y):
        """
        Internal fit method for sklearn-compatible data formats.
        """
        return self

    def _fit_numpy(self, X, y):
        """
        Internal fit method for sklearn-compatible data formats.
        """
        return self

    def _transform_pandas(self, X):
        """
        Internal predict method for sklearn-compatible data formats.
        """
        return X

    def _transform_arrow(self, X):
        """
        Internal predict method for sklearn-compatible data formats.
        """
        return X

    def _transform_polars(self, X):
        """
        Internal predict method for sklearn-compatible data formats.
        """
        return X

    def _transform_numpy(self, X):
        """
        Internal predict method for sklearn-compatible data formats.
        """
        return X


class TestMockModel(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, count_data, sample_metadata):
        self.sample_metadata = sample_metadata
        self.metadata_columns = list(sample_metadata.columns)
        self.X, self.y = count_data
        self.column_names = list(self.X.columns)
        self.target_column = self.y.columns[0]
        self.data = pd.concat([self.sample_metadata, self.X, self.y], axis=1)

    def setUp(self):
        self.params = {
            "version": version.__version__,
            "processor_param_int": 1,
            "processor_param_float": 1.0,
            "processor_param_str": "test",
            "processor_param_bool": True,
            "processor_param_list": [1, 2, 3],
            "processor_param_dict": {"a": 1, "b": 2},
            "processor_param_tuple": (1, 2),
        }
        self.config = MockProcessorConfig(**self.params)
        self.model = MockModel(config=self.config)
        self.preprocessor = MockPreprocessor(config=self.config)
        self.funcs = [
            self.preprocessor._fit_pandas,
            self.preprocessor._fit_numpy,
            self.preprocessor._fit_arrow,
            self.preprocessor._fit_polars,
        ]
        self.accepted_formats = ["pandas", "numpy", "arrow", "polars"]

    def tearDown(self):
        # clean up the cache directory
        cache_dir = biofit.config.BIOFIT_CACHE_HOME
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    def test_init(self):
        model_params = self.model.config.get_params()
        preprocessor_params = self.preprocessor.config.get_params()
        self.assertEqual(model_params, preprocessor_params)
        self.assertEqual(model_params, self.params)
        self.assertEqual(preprocessor_params, self.params)

    def test_get_method(self):
        formats = ["numpy", "pandas", "arrow", "polars"]
        func_type = "_fit"
        methods = self.preprocessor._get_method(formats, func_type)
        method_names = [method.__name__ for method in methods]
        expected_method_names = [
            "_fit_numpy",
            "_fit_pandas",
            "_fit_arrow",
            "_fit_polars",
        ]
        self.assertEqual(method_names, expected_method_names)

    def test_has_method(self):
        formats = ["numpy", "pandas", "arrow", "polars"]
        func_type = "_fit"
        has_method = self.preprocessor._has_method(formats, func_type)
        self.assertTrue(has_method)
        formats = ["nonexistent_format"]
        has_method = self.preprocessor._has_method(formats, func_type)
        self.assertFalse(has_method)

    def test_get_target_func_match_source_format(self):
        # Testing _get_target_func method
        # Priority should be source format
        func, to_format = self.preprocessor._get_target_func(
            funcs=self.funcs,
            source_format="numpy",
            accepted_formats=self.accepted_formats,
        )

        self.assertEqual(func.__name__, "_fit_numpy")
        self.assertEqual(to_format, "numpy")

    def test_get_target_func_priority(self):
        # Priority should be the first format found in funcs in the order of
        # accepted_formats
        func, to_format = self.preprocessor._get_target_func(
            funcs=self.funcs[-2:],
            source_format="not_a_format",
            accepted_formats=self.accepted_formats,
        )

        self.assertEqual(func.__name__, f"_fit_{self.accepted_formats[-2]}")
        self.assertEqual(to_format, self.accepted_formats[-2])

    def test_get_target_func_priority_order(self):
        # Priority is the first format found in funcs in the order of
        # target_formats
        func, to_format = self.preprocessor._get_target_func(
            funcs=self.funcs,
            source_format="numpy",
            target_formats=["pandas", "arrow"],
        )
        self.assertEqual(func.__name__, "_fit_pandas")
        self.assertEqual(to_format, "pandas")

    def test_set_input_columns_and_arity(self):
        result = self.preprocessor._set_input_columns_and_arity(
            "col1", ["col2", "col3"]
        )
        expected = [["col1"], ["col2", "col3"]]
        self.assertEqual(result, expected)

    def test_reinsert_columns_valid(self):
        """Test reinsert_columns with valid inputs and unused indices."""
        # Input data (input)
        input_data = pd.DataFrame(
            {
                "col1": np.arange(10),
                "col2": np.arange(10, 20),
                "col3": np.arange(20, 30),
                "col4": np.arange(30, 40),
            }
        )
        # Output data (out)
        output_data = pd.DataFrame({"col1_transformed": np.arange(100, 110)})
        # Indices of unused columns (unused_indices)
        indices = [0, 3]  # 'col1' and 'col4' transformed into 'col1_transformed'
        unused_indices = [1, 2]  # 'col2' and 'col3' were unused

        self.preprocessor.config._n_features_out = 1
        result = self.preprocessor._reinsert_columns(
            input=input_data,
            out=output_data,
            indices=indices,
            unused_indices=unused_indices,
            one_to_one_features=self.preprocessor.config.one_to_one_features,
        )
        self.preprocessor.config._n_features_out = None

        # When one_to_one_features=False, the output columns should be
        # appended to the end of the input data
        # Expected result: 'col2', 'col3', and 'col1_transformed'
        other_cols = input_data.iloc[:, unused_indices]
        concatenated = pd.concat([output_data, other_cols], axis=1)
        expected_output = concatenated[["col2", "col3", "col1_transformed"]]

        # Assert that the result matches the expected output
        pd.testing.assert_frame_equal(result, expected_output)

    def test_reinsert_columns_one_to_one_features(self):
        """Test reinsert_columns with one_to_one_features=True."""
        # Input data
        input_data = pd.DataFrame(
            {
                "col0": np.arange(10),
                "col1": np.arange(10, 20),
                "col2": np.arange(20, 30),
                "col3": np.arange(30, 40),
            }
        )
        # Output data
        output_data = pd.DataFrame(
            {
                "col0_transformed": np.arange(100, 110),
                "col3_transformed": np.arange(200, 210),
            }
        )
        # Indices used
        indices = [0, 3]  # 'col0' and 'col3' transformed
        # Unused indices
        unused_indices = [1, 2]  # 'col1' and 'col2' unused

        self.preprocessor.config._n_features_out = None

        # when one_to_one_features=True, the output columns should be follow
        # the same order as the input columns
        result = self.preprocessor._reinsert_columns(
            input=input_data,
            out=output_data,
            indices=indices,
            unused_indices=unused_indices,
            one_to_one_features=self.preprocessor.config.one_to_one_features,
        )

        # Expected result
        other_cols = input_data.iloc[:, unused_indices]
        expected_output = pd.concat([other_cols, output_data], axis=1)
        expected_output = expected_output[
            ["col0_transformed", "col1", "col2", "col3_transformed"]
        ]

        # Assert
        pd.testing.assert_frame_equal(result, expected_output)

    def test_reinsert_columns_no_unused_columns(self):
        """Test when there are no unused columns."""
        input_data = pd.DataFrame({"col1": np.arange(10)})
        output_data = pd.DataFrame({"col1_transformed": np.arange(100, 110)})
        indices = [0]
        unused_indices = []

        result = self.preprocessor._reinsert_columns(
            input_data, output_data, indices, unused_indices
        )

        # Expected result is the same as output_data
        pd.testing.assert_frame_equal(result, output_data)

    def test_reinsert_columns_mismatched_row_counts(self):
        """Test when input and output have different number of rows."""
        input_data = pd.DataFrame({"col1": np.arange(10)})
        output_data = pd.DataFrame(
            {
                "col1_transformed": np.arange(5)  # Different number of rows
            }
        )
        indices = [0]
        unused_indices = []

        result = self.preprocessor._reinsert_columns(
            input_data, output_data, indices, unused_indices
        )

        # Since row counts differ, should return output as is
        pd.testing.assert_frame_equal(result, output_data)

    def test_reinsert_columns_invalid_indices(self):
        """Test with invalid indices (out of bounds)."""
        input_data = pd.DataFrame({"col1": np.arange(10)})
        output_data = pd.DataFrame({"col1_transformed": np.arange(10)})
        indices = [0]
        unused_indices = [1]  # Invalid index

        with self.assertRaises(IndexError):
            self.preprocessor._reinsert_columns(
                input_data, output_data, indices, unused_indices
            )

    def test_reinsert_columns_empty_input(self):
        """Test with empty input data."""
        input_data = pd.DataFrame()
        output_data = pd.DataFrame({"col_transformed": []})
        indices = []
        unused_indices = []

        result = self.preprocessor._reinsert_columns(
            input_data, output_data, indices, unused_indices
        )

        # Expected result is the same as output_data
        pd.testing.assert_frame_equal(result, output_data)

    def test_reinsert_columns_empty_output(self):
        """Test with empty output data and unused columns."""
        input_data = pd.DataFrame({"col1": [], "col2": []})
        output_data = pd.DataFrame()
        indices = []
        unused_indices = [0, 1]

        result = self.preprocessor._reinsert_columns(
            input_data, output_data, indices, unused_indices
        )

        # Expected result is input_data with unused columns
        pd.testing.assert_frame_equal(result, input_data)

    def test_reinsert_columns_different_column_types(self):
        """Test with different data types in columns."""
        input_data = pd.DataFrame(
            {
                "int_col": np.arange(10),
                "float_col": np.random.rand(10),
                "str_col": ["text"] * 10,
            }
        )
        output_data = pd.DataFrame({"int_col_transformed": np.arange(100, 110)})
        indices = [0]
        unused_indices = [1, 2]

        result = self.preprocessor._reinsert_columns(
            input_data, output_data, indices, unused_indices, one_to_one_features=True
        )

        other_cols = input_data[["float_col", "str_col"]]
        expected_output = pd.concat([output_data, other_cols], axis=1)
        expected_output = expected_output[
            ["int_col_transformed", "float_col", "str_col"]
        ]

        pd.testing.assert_frame_equal(result, expected_output)

    def test_reinsert_columns_null_values(self):
        """Test with null values in input data."""
        input_data = pd.DataFrame(
            {
                "col1": [1, np.nan, 3, np.nan, 5],
                "col2": [np.nan, 2, np.nan, 4, np.nan],
            }
        )
        output_data = pd.DataFrame({"col1_transformed": [10, 20, 30, 40, 50]})
        indices = [0]
        unused_indices = [1]

        result = self.preprocessor._reinsert_columns(
            input_data, output_data, indices, unused_indices, one_to_one_features=True
        )

        other_cols = input_data[["col2"]]
        expected_output = pd.concat([output_data, other_cols], axis=1)
        expected_output = expected_output[["col1_transformed", "col2"]]

        pd.testing.assert_frame_equal(result, expected_output)

    def test_reinsert_columns_non_dataframe_input(self):
        """Test with input data that is not a DataFrame."""
        input_data = pd.DataFrame(
            {
                "col1": np.arange(10),
                "col2": np.arange(10, 20),
                "col3": np.arange(20, 30),
            }
        )
        output_data = np.arange(10)
        indices = [0]
        unused_indices = [1, 2]

        output_data = self.preprocessor._reinsert_columns(
            input_data, output_data, indices, unused_indices
        )
        # format type should match the input data
        self.assertIsInstance(output_data, pd.DataFrame)

    def test_make_columns_exclusive(self):
        columns = [["col1", "col2"], ["col2", "col3"], ["col3", "col4"]]
        result = self.preprocessor._make_columns_exclusive(columns)
        expected = [["col1"], ["col2"], ["col3", "col4"]]
        self.assertEqual(result, expected)

    def test_get_columns_valid_input_columns(self):
        """Test _get_columns with valid input_columns"""
        input_columns = self.column_names
        result = self.preprocessor._get_columns(
            self.X, input_columns=[input_columns], raise_if_missing=True
        )
        # Unpack results
        (
            feature_names_in,
            feature_idx_in,
            unused_idx_in,
            extra_names_in,
            extra_idx_in,
            unused_extra_idx_in,
            offsets,
        ) = result

        # Assertions
        self.assertEqual(feature_names_in, input_columns)
        expected_indices = [self.column_names.index(col) for col in input_columns]
        self.assertEqual(feature_idx_in, expected_indices)
        expected_unused_indices = sorted(
            set(range(len(self.column_names))) - set(expected_indices)
        )
        self.assertEqual(unused_idx_in, expected_unused_indices)
        self.assertIsNone(extra_names_in)
        self.assertIsNone(extra_idx_in)
        self.assertIsNone(unused_extra_idx_in)
        self.assertIsNone(offsets)

    def test_get_columns_invalid_input_columns_raise(self):
        """Test _get_columns with invalid input_columns and raise_if_missing=True"""
        input_columns = self.column_names + ["non_existent_column"]
        with self.assertRaises(ValueError) as context:
            self.preprocessor._get_columns(
                self.X, input_columns=[input_columns], raise_if_missing=True
            )
        self.assertIn(
            str(context.exception),
            "Columns {'non_existent_column'} not found in input dataset",
        )

    def test_get_columns_invalid_input_columns_no_raise(self):
        """Test _get_columns with invalid input_columns and raise_if_missing=False"""
        input_columns = self.column_names + ["non_existent_column"]
        result = self.preprocessor._get_columns(
            self.X, input_columns=[input_columns], raise_if_missing=False
        )
        (
            feature_names_in,
            feature_idx_in,
            unused_idx_in,
            extra_names_in,
            extra_idx_in,
            unused_extra_idx_in,
            offsets,
        ) = result

        self.assertEqual(feature_names_in, self.column_names)
        expected_indices = list(range(len(self.column_names)))
        self.assertEqual(feature_idx_in, expected_indices)
        expected_unused_indices = sorted(
            set(range(len(self.column_names))) - set(expected_indices)
        )
        self.assertEqual(unused_idx_in, expected_unused_indices)
        self.assertIsNone(extra_names_in)
        self.assertIsNone(extra_idx_in)
        self.assertIsNone(unused_extra_idx_in)
        self.assertIsNone(offsets)

    def test_get_columns_unused_columns(self):
        """Test _get_columns with unused_columns specified"""
        unused_columns = self.column_names[:2]
        result = self.preprocessor._get_columns(
            self.X,
            input_columns=[None],
            unused_columns=[unused_columns],
            raise_if_missing=True,
        )
        # Unpack results
        (
            feature_names_in,
            feature_idx_in,
            unused_idx_in,
            extra_names_in,
            extra_idx_in,
            unused_extra_idx_in,
            offsets,
        ) = result

        expected_input_columns = [
            col for col in self.column_names if col not in unused_columns
        ]
        self.assertEqual(feature_names_in, expected_input_columns)
        expected_indices = [
            self.column_names.index(col) for col in expected_input_columns
        ]
        self.assertEqual(feature_idx_in, expected_indices)
        expected_unused_indices = sorted(
            [self.column_names.index(col) for col in unused_columns]
        )
        self.assertEqual(unused_idx_in, expected_unused_indices)
        self.assertIsNone(extra_names_in)
        self.assertIsNone(extra_idx_in)
        self.assertIsNone(unused_extra_idx_in)
        self.assertIsNone(offsets)

    def test_get_columns_with_args(self):
        """Test _get_columns with additional args (extra inputs)"""
        # Extra input data
        extra_X = {
            "extra_feature_1": np.random.rand(50),
            "extra_feature_2": np.random.rand(50),
        }
        input_columns = [self.column_names, ["extra_feature_1"]]
        result = self.preprocessor._get_columns(
            self.X, extra_X, input_columns=input_columns, raise_if_missing=True
        )
        # Unpack results
        (
            feature_names_in,
            feature_idx_in,
            unused_idx_in,
            extra_names_in,
            extra_idx_in,
            unused_extra_idx_in,
            offsets,
        ) = result

        # Assertions for main input
        self.assertEqual(feature_names_in, self.column_names)
        expected_indices = list(range(len(self.column_names)))
        self.assertEqual(feature_idx_in, expected_indices)
        expected_unused_indices = sorted(
            set(range(len(self.column_names))) - set(expected_indices)
        )
        self.assertEqual(unused_idx_in, expected_unused_indices)

        # Assertions for extra input
        self.assertEqual(extra_names_in, [["extra_feature_1"]])
        self.assertEqual(extra_idx_in, [[0]])
        self.assertEqual(unused_extra_idx_in, [[1]])  # Only 'extra_feature_2' is unused
        self.assertEqual(offsets, [])  # No offsets since extra input not concatenated

    def test_get_columns_empty_input_columns(self):
        """Test _get_columns with empty input_columns"""
        result = self.preprocessor._get_columns(self.X, input_columns=[])
        self.assertEqual(result, (None, None, None, None, None, None, None))

    def test_get_columns_input_feature_types(self):
        """Test _get_columns with input_feature_types specified"""
        input_feature_types = get_feature("Abundance")
        result = self.preprocessor._get_columns(
            self.X,
            input_columns=[None],
            input_feature_types=[input_feature_types],
            raise_if_missing=True,
        )
        # Unpack results
        (
            feature_names_in,
            feature_idx_in,
            unused_idx_in,
            extra_names_in,
            extra_idx_in,
            unused_extra_idx_in,
            offsets,
        ) = result

        # Assertions
        self.assertEqual(feature_names_in, self.column_names)
        expected_indices = list(range(len(self.column_names)))
        self.assertEqual(feature_idx_in, expected_indices)
        expected_unused_indices = sorted(
            set(range(len(self.column_names))) - set(expected_indices)
        )
        self.assertEqual(unused_idx_in, expected_unused_indices)
        self.assertIsNone(extra_names_in)
        self.assertIsNone(extra_idx_in)
        self.assertIsNone(unused_extra_idx_in)
        self.assertIsNone(offsets)

    @require_biosets
    def test_get_columns_invalid_input_feature_types(self):
        """Test _get_columns with invalid input_feature_types"""

        X = create_bioset(
            self.X,
            feature_type=get_feature("Abundance"),
        )

        # Dataset uses Abundance feature type, but GenomicVariant is specified
        result = self.preprocessor._get_columns(
            X,
            input_columns=[None],
            input_feature_types=[get_feature("GenomicVariant")],
            raise_if_missing=False,
        )
        # Should return all columns since it doesn't match the feature type
        self.assertEqual(
            result,
            (
                ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"],
                [0, 1, 2, 3, 4],
                [],
                None,
                None,
                None,
                None,
            ),
        )

    def test_get_columns_mismatched_arity(self):
        """Test _get_columns with mismatched arity between input_columns and args"""
        # No args provided, but input_columns has two lists
        input_columns = [self.column_names, ["extra_feature_1"]]
        with self.assertRaises(AssertionError) as context:
            self.preprocessor._get_columns(
                self.X, input_columns=input_columns, raise_if_missing=True
            )
        self.assertIn(
            str(context.exception),
            "Number of column sets (2) must match the arity (1)",
        )

    def test_get_columns_with_none_X(self):
        """Test _get_columns with X as None"""
        with self.assertRaises(AssertionError) as context:
            self.preprocessor._get_columns(
                None, input_columns=[self.column_names], raise_if_missing=True
            )
        self.assertIn("Input data is None", str(context.exception))

    def test_get_columns_with_non_list_input_columns(self):
        """Test _get_columns with input_columns not wrapped in a list"""
        input_columns = "feature_1"
        with self.assertRaises(AssertionError) as context:
            self.preprocessor._get_columns(
                self.X, input_columns=input_columns, raise_if_missing=True
            )
        # Should return None values since input_columns is not properly wrapped
        self.assertIn(
            str(context.exception),
            f"input_columns must be a list of column names or indices, "
            f"but got {type(input_columns)}",
        )

    @require_biosets
    def test_get_columns_with_unused_feature_types(self):
        """Test _get_columns with unused_feature_types specified"""
        X = create_bioset(
            self.X,
            self.y,
            self.sample_metadata,
            feature_type=get_feature("Abundance"),
            target_type=get_feature("BinClassLabel"),
        )
        unused_feature_types = get_feature("METADATA_FEATURE_TYPES")
        result = self.preprocessor._get_columns(
            X,
            input_columns=[None],
            unused_feature_types=[unused_feature_types],
            raise_if_missing=True,
        )
        # Unpack results
        (
            feature_names_in,
            feature_idx_in,
            unused_idx_in,
            _,
            _,
            _,
            _,
        ) = result

        self.assertEqual(feature_names_in, self.column_names)
        expected_indices = [X.column_names.index(col) for col in self.column_names]
        self.assertEqual(feature_idx_in, expected_indices)
        expected_unused_indices = sorted(
            [
                X.column_names.index(col)
                for col in self.metadata_columns + [self.target_column]
            ]
        )
        self.assertEqual(unused_idx_in, expected_unused_indices)

    def test_get_columns_with_all_none_parameters(self):
        """Test _get_columns with all parameters as None"""
        result = self.preprocessor._get_columns(
            self.X,
            input_columns=None,
            input_feature_types=None,
            unused_columns=None,
            unused_feature_types=None,
            raise_if_missing=False,
        )
        # Should return None values
        self.assertEqual(result, (None, None, None, None, None, None, None))

    def test_get_columns_with_args_none(self):
        """Test _get_columns with args as None"""
        result = self.preprocessor._get_columns(
            self.data,
            None,
            input_columns=[
                self.metadata_columns,
                self.column_names,
            ],
            raise_if_missing=True,
        )
        # Unpack results
        (
            feature_names_in,
            feature_idx_in,
            _,
            extra_names_in,
            extra_idx_in,
            unused_extra_idx_in,
            offsets,
        ) = result
        # Assertions for main input
        self.assertEqual(feature_names_in, self.metadata_columns)
        cols = list(self.data.columns)
        expected_indices = [cols.index(col) for col in self.metadata_columns]
        self.assertEqual(feature_idx_in, expected_indices)

        # Assertions for extra input (args)
        self.assertEqual(extra_names_in, [self.column_names])
        self.assertEqual(
            extra_idx_in,
            [[cols.index(col) for col in self.column_names]],
        )
        self.assertEqual(unused_extra_idx_in, None)
        self.assertEqual(offsets, [0])

    def test_get_columns_with_args_mismatched_rows(self):
        """Test _get_columns with args having mismatched number of rows"""
        # Extra input data with different number of rows
        extra_X = {
            "extra_feature_1": np.random.rand(self.data.shape[0] + 1),
        }
        input_columns = [self.column_names, ["extra_feature_1"]]
        result = self.preprocessor._get_columns(
            self.X, extra_X, input_columns=input_columns, raise_if_missing=True
        )
        # Since row numbers don't match, offsets should not be incremented
        (_, _, _, _, _, _, offsets) = result
        self.assertEqual(offsets, [])  # No offsets should be added

    def test_generate_fingerprint(self):
        fingerprint = "initial_fingerprint"
        generated_fp = self.preprocessor.generate_fingerprint(fingerprint, self.config)
        self.assertIsNotNone(generated_fp)
        self.assertIsInstance(generated_fp, str)

    def test_from_config(self):
        new_processor = MockPreprocessor._from_config(self.config)
        self.assertIsInstance(new_processor, MockPreprocessor)
        self.assertEqual(new_processor.config, self.config)

    def test_call(self):
        # for ray-tune compatibility
        batch = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        kwargs = {
            "fn": self.preprocessor._transform_pandas,
            "func_type": "_transform",
            "selected_indices": [0, 1],
            "unused_indices": [2, 3],
            "keep_unused_columns": False,
            "in_format_kwargs": {"target_format": "pandas"},
            "out_format_kwargs": {"target_format": "pandas"},
        }
        result = self.preprocessor(batch, **kwargs)
        pd.testing.assert_frame_equal(result, batch)

    def test_set_params(self):
        self.preprocessor.set_params(
            processor_param_int=99,
            processor_param_float=99.0,
            processor_param_str="new",
            processor_param_bool=False,
            processor_param_list=[99, 99, 99],
            processor_param_dict={"a": 99, "b": 99},
            processor_param_tuple=(99, 99),
        )
        self.assertEqual(self.preprocessor.config.processor_param_int, 99)
        self.assertEqual(self.preprocessor.config.processor_param_float, 99.0)
        self.assertEqual(self.preprocessor.config.processor_param_str, "new")
        self.assertEqual(self.preprocessor.config.processor_param_bool, False)
        self.assertEqual(self.preprocessor.config.processor_param_list, [99, 99, 99])
        self.assertEqual(
            self.preprocessor.config.processor_param_dict, {"a": 99, "b": 99}
        )
        self.assertEqual(self.preprocessor.config.processor_param_tuple, (99, 99))

    def test_is_fitted(self):
        self.assertFalse(self.preprocessor.is_fitted)
        self.preprocessor.fit(self.X, self.y)
        self.assertTrue(self.preprocessor.is_fitted)

    def test_has_fit(self):
        self.assertTrue(self.preprocessor.has_fit)

    def test_process_fit_without_input_columns(self):
        # Test _process_fit without input_columns
        self.assertFalse(self.model.is_fitted)
        with self.assertRaises(AssertionError) as context:
            self.preprocessor._process_fit(self.X, self.y)
        self.assertIn(
            str(context.exception),
            "The `fit` method of `MockPreprocessor` must call:\n"
            "```\n"
            "self.config._input_columns = self._set_input_columns_and_arity(*args)"
            "\n```\n"
            "Where `*args` are the columns for each input dataset.",
        )
        with self.assertRaises(AssertionError) as context:
            self.model._process_fit(self.X, self.y)
        self.assertIn(
            str(context.exception),
            "The `fit` method of `MockModel` must call:\n"
            "```\n"
            "self.config._input_columns = self._set_input_columns_and_arity(*args)"
            "\n```\n"
            "Where `*args` are the columns for each input dataset.",
        )

    def test_process_transform_without_input_columns(self):
        # Test _process_fit without input_columns
        with self.assertRaises(AssertionError) as context:
            self.preprocessor._process_transform(self.X)
        self.assertIn(
            str(context.exception),
            "The `transform` method of `MockPreprocessor` must call:\n"
            "```\n"
            "self._input_columns = self._set_input_columns_and_arity(*args)"
            "\n```\n"
            "Where `*args` are the columns for each input dataset.",
        )

        with self.assertRaises(AssertionError) as context:
            self.model._method_prefix = "_predict"
            self.model._process_transform(self.X)
        self.assertIn(
            str(context.exception),
            "The `predict` method of `MockModel` must call:\n"
            "```\n"
            "self._input_columns = self._set_input_columns_and_arity(*args)"
            "\n```\n"
            "Where `*args` are the columns for each input dataset.",
        )

        with self.assertRaises(AssertionError) as context:
            self.model._method_prefix = "_predict_proba"
            self.model._process_transform(self.X)
        self.assertIn(
            str(context.exception),
            "The `predict_proba` method of `MockModel` must call:\n"
            "```\n"
            "self._input_columns = self._set_input_columns_and_arity(*args)"
            "\n```\n"
            "Where `*args` are the columns for each input dataset.",
        )

    def test_process_fit_with_absolute_path_cache_file_name(self):
        # make cache_file_name an absolute path
        cache_file_name = (
            biofit.config.BIOFIT_PROCESSORS_CACHE / "cache.json"
        ).as_posix()

        self.assertFalse(self.model.is_fitted)
        with self.assertRaises(ValueError) as context:
            self.model.config._input_columns = self.model._set_input_columns_and_arity(
                None, None
            )
            self.model._process_fit(self.X, self.y, cache_file_name=cache_file_name)
            delattr(self.model.config, "_input_columns")

        self.assertIn(
            str(context.exception),
            "`cache_file_name` is an absolute path. Please provide the "
            "file name only. You can specify the directory using "
            "`cache_dir`.",
        )

    def test_process_fit_with_remote_cache_file_name(self):
        # make cache_file_name an absolute path
        cache_file_name = "s3://bucket/cache.json"

        self.assertFalse(self.model.is_fitted)
        with self.assertRaises(ValueError) as context:
            self.model.config._input_columns = self.model._set_input_columns_and_arity(
                None, None
            )
            self.model._process_fit(self.X, self.y, cache_file_name=cache_file_name)
            delattr(self.model.config, "_input_columns")

        self.assertIn(
            str(context.exception),
            "`cache_file_name` is a remote URL. Please provide the "
            "file name only. You can specify the directory using "
            "`cache_dir`.",
        )

    def test_process_tranform_with_absolute_path_cache_file_name(self):
        # make cache_file_name an absolute path
        cache_file_name = (
            biofit.config.BIOFIT_PROCESSORS_CACHE / "cache.json"
        ).as_posix()

        with self.assertRaises(ValueError) as context:
            self.model._input_columns = self.model._set_input_columns_and_arity(
                None, None
            )
            self.model._process_transform(self.X, cache_file_name=cache_file_name)
            delattr(self.model, "_input_columns")

        self.assertIn(
            str(context.exception),
            "`cache_file_name` is an absolute path. Please provide the "
            "file name only. You can specify the directory using "
            "`cache_dir`.",
        )

    def test_process_transform_with_remote_cache_file_name(self):
        # make cache_file_name an absolute path
        cache_file_name = "s3://bucket/cache.json"

        with self.assertRaises(ValueError) as context:
            self.model._input_columns = self.model._set_input_columns_and_arity(
                None, None
            )
            self.model._process_transform(self.X, cache_file_name=cache_file_name)
            delattr(self.model, "_input_columns")

        self.assertIn(
            str(context.exception),
            "`cache_file_name` is a remote URL. Please provide the "
            "file name only. You can specify the directory using "
            "`cache_dir`.",
        )

    def test_load_processed_estimator_from_cache_invalid_file(self):
        # Use a non-existent cache file path
        cache_file_name = "/non/existent/cache/file"

        # Expect NonExistentCacheError to be raised
        with self.assertRaises(NonExistentCacheError):
            self.model.load_processed_estimator_from_cache(cache_file_name)

    def test_process_fit_without_loading_from_cache(self):
        # Set up the processor with caching enabled
        self.model.config.enable_caching = True
        self.model.config.cache_output = True
        self.model.config.load_from_cache_file = False
        self.assertFalse(self.model.is_fitted)

        # Mock the load_processed_estimator_from_cache method to raise NonExistentCacheError
        with patch.object(
            self.model,
            "load_processed_estimator_from_cache",
            side_effect=NonExistentCacheError,
        ) as mock_load_cache:
            with patch.object(
                self.model.config,
                "save_to_cache",
                wraps=self.model.config.save_to_cache,
            ) as mock_save_cache:
                # Mock the _fit method to confirm it is called
                # Call _process_fit
                self.model.config._input_columns = (
                    self.model._set_input_columns_and_arity(None, None)
                )
                self.model._process_fit(
                    self.X,
                    self.y,
                    cache_file_name="cache.json",
                    fingerprint="test",
                )

                mock_load_cache.assert_not_called()
                mock_save_cache.assert_called_once()

                cache_dir = generate_cache_dir(
                    self.model,
                    self.model.config._data_fingerprint,
                    root_dir=biofit.config.BIOFIT_PROCESSORS_CACHE,
                )
                cache_path = Path(cache_dir) / "cache.json"
                self.assertTrue(cache_path.exists())

        # Check that the model is marked as fitted
        self.assertTrue(self.model.is_fitted)

    def test_process_fit_without_saving_output_to_cache(self):
        self.assertFalse(self.model.is_fitted)

        # Run it once to save the cache
        self.model.config._input_columns = self.model._set_input_columns_and_arity(
            None, None
        )
        self.model._process_fit(
            self.X,
            self.y,
            cache_file_name="cache.json",
            fingerprint="test",
        )

        # Set up the processor with only loading from cache enabled
        self.model.config.enable_caching = True
        self.model.config.cache_output = False
        self.model.config.load_from_cache_file = True

        # Also check if fingerprint is consistent
        old_data_fingerprint = self.model.config._data_fingerprint
        old_processor_fingerprint = self.model.fingerprint

        with patch.object(
            self.model,
            "load_processed_estimator_from_cache",
            side_effect=NonExistentCacheError,
        ) as mock_load_cache:
            with patch.object(
                self.model.config,
                "save_to_cache",
                wraps=self.model.config.save_to_cache,
            ) as mock_save_cache:
                # Run it again to load from cache
                self.model.config._input_columns = (
                    self.model._set_input_columns_and_arity(None, None)
                )
                self.model._process_fit(
                    self.X,
                    self.y,
                    cache_file_name="cache.json",
                    fingerprint="test",
                )

                # Check to see if fingerprint is consistent
                self.assertEqual(
                    old_data_fingerprint, self.model.config._data_fingerprint
                )
                self.assertEqual(old_processor_fingerprint, self.model.fingerprint)

                cache_dir = generate_cache_dir(
                    self.model,
                    self.model.config._data_fingerprint,
                    root_dir=biofit.config.BIOFIT_PROCESSORS_CACHE,
                )
                cache_path = Path(cache_dir) / "cache.json"

                # Load should be called but save should not
                mock_load_cache.assert_called_once_with(cache_path.as_posix())
                mock_save_cache.assert_not_called()

                self.assertTrue(cache_path.exists())

        # Check that the model is marked as fitted
        self.assertTrue(self.model.is_fitted)

    @require_datasets
    def test_transform_output_consistency_with_and_without_cache(self):
        # Fit the model first without caching
        X = DataHandler.to_dataset(self.X)
        y = DataHandler.to_dataset(self.y)
        original_fingerprint = X._fingerprint
        original_target_fingerprint = y._fingerprint
        with patch.object(
            self.model, "_fit_sklearn", wraps=self.model._fit_sklearn
        ) as mock_fit:
            self.model._fit_sklearn.__name__ = "_fit_sklearn"
            self.model.fit(X, y)

        # Make sure the fingerprint didn't change after fit
        self.assertEqual(original_target_fingerprint, y._fingerprint)
        self.assertEqual(original_target_fingerprint, "643b0a98d71c180a")

        mock_fit.assert_called_once()
        output_no_cache = self.model.predict(X)

        # Calculate the fingerprint of the output dataset without cache
        fingerprint_no_cache = output_no_cache._fingerprint
        with patch.object(
            self.model, "_fit_sklearn", wraps=self.model._fit_sklearn
        ) as mock_fit:
            # Fit the model again (should create cache files)
            self.model._fit_sklearn.__name__ = "_fit_sklearn"
            self.model.fit(X, y)

        # Since loaded from cache, fit should not be called
        mock_fit.assert_not_called()

        # Transform the data with cache enabled
        output_with_cache = self.model.predict(X)

        # Calculate the fingerprint of the output dataset with cache
        fingerprint_with_cache = output_with_cache._fingerprint

        # Transform a separate data with cache enabled
        other_output_with_cache = self.model.predict(y)

        other_fingerprint_with_cache = other_output_with_cache._fingerprint

        # Compare the outputs
        self.assertEqual(len(output_no_cache), len(output_with_cache))
        output_no_cache = DataHandler.to_pandas(output_no_cache)
        output_with_cache = DataHandler.to_pandas(output_with_cache)
        pd.testing.assert_frame_equal(output_no_cache, output_with_cache)
        # Compare the fingerprints
        self.assertEqual(original_fingerprint, X._fingerprint)
        self.assertEqual(original_fingerprint, "d89db52fb3a40be9")
        self.assertEqual(fingerprint_no_cache, fingerprint_with_cache)
        self.assertEqual(fingerprint_no_cache, "91acea130eb575e8")
        self.assertNotEqual(fingerprint_no_cache, other_fingerprint_with_cache)
        self.assertEqual(other_fingerprint_with_cache, "950ec73181f227a8")

    def test_parse_fingerprint(self):
        fingerprint = "base_fingerprint"
        parsed_fp = self.preprocessor._parse_fingerprint(fingerprint)
        self.assertIn(fingerprint, parsed_fp)
        self.assertIn(self.config.processor_name, parsed_fp)

    def test_reset(self):
        # Test if the reset method resets the processor to the state before the first
        # fit was called
        preprocessor = copy.deepcopy(self.preprocessor.fit(self.X, self.y))

        # Change the parameters of the processor
        preprocessor.config.processor_param_int += 10
        preprocessor.config.processor_param_float = 10.0

        # Reset the processor
        preprocessor._reset(preprocessor.config_)

        # Should be the same as the original processor
        self.assertEqual(
            preprocessor.config.get_params(), self.preprocessor.config.get_params()
        )

    def test_from_config_classmethod(self):
        new_processor = MockPreprocessor.from_config(self.config)
        self.assertIsInstance(new_processor, MockPreprocessor)
        self.assertEqual(new_processor.config, self.config)

    def test_validate_fit_params_input_raise(self):
        self.preprocessor.config._fit_input_feature_types = [get_feature("Abundance")]
        self.preprocessor.config._fit_unused_feature_types = None
        self.preprocessor.config._transform_input_feature_types = None
        self.preprocessor.config._transform_unused_feature_types = None
        self.preprocessor.config._input_columns = [None, None]
        with self.assertRaises(AssertionError) as context:
            self.preprocessor._validate_fit_params(2)
        self.assertIn(
            str(context.exception),
            "`_fit_input_feature_types` is defined in "
            "MockProcessorConfig but does not match the arity of "
            "the fit function in MockPreprocessor (i.e. len("
            "self.config._fit_input_feature_types) != "
            "len(self.config._input_columns) -> "
            "1 != 2).\n"
            "This can be corrected by doing, for example:\n"
            "_fit_input_feature_types = field(\n"
            "    default_factory=lambda: [None, None], init=False, "
            "repr=False\n"
            ")",
        )

    def test_validate_fit_params_unused_raise(self):
        self.preprocessor.config._fit_input_feature_types = None
        self.preprocessor.config._fit_unused_feature_types = [get_feature("Abundance")]
        self.preprocessor.config._transform_input_feature_types = None
        self.preprocessor.config._transform_unused_feature_types = None
        self.preprocessor.config._input_columns = [None, None]
        with self.assertRaises(AssertionError) as context:
            self.preprocessor._validate_fit_params(2)
        self.assertIn(
            str(context.exception),
            "`_fit_unused_feature_types` is defined in "
            "MockProcessorConfig but does not match the arity of "
            "the fit function in MockPreprocessor (i.e. len("
            "self.config._fit_unused_feature_types) != "
            "len(self.config._input_columns) -> "
            "1 != 2).\n"
            "This can be corrected by doing, for example:\n"
            "_fit_unused_feature_types = field(\n"
            "    default_factory=lambda: [None, None], init=False, "
            "repr=False\n"
            ")",
        )

    def test_validate_transform_params_input_raise(self):
        self.preprocessor.config._fit_input_feature_types = None
        self.preprocessor.config._fit_unused_feature_types = None
        self.preprocessor.config._transform_input_feature_types = [
            get_feature("Abundance")
        ]
        self.preprocessor.config._transform_unused_feature_types = None
        self.preprocessor._input_columns = [None, None]
        with self.assertRaises(AssertionError) as context:
            self.preprocessor._validate_transform_params(2)
        self.assertIn(
            str(context.exception),
            "`_transform_input_feature_types` is defined in "
            "MockProcessorConfig but does not match the arity of "
            "the transform function in MockPreprocessor (i.e. len("
            "self.config._transform_input_feature_types) != "
            "len(self._input_columns) -> "
            "1 != 2).\n"
            "This can be corrected by doing, for example:\n"
            "_transform_input_feature_types = field(\n"
            "    default_factory=lambda: [None, None], init=False, "
            "repr=False\n"
            ")",
        )

    def test_validate_transform_params_unused_raise(self):
        self.preprocessor.config._fit_input_feature_types = None
        self.preprocessor.config._fit_unused_feature_types = None
        self.preprocessor.config._transform_input_feature_types = None
        self.preprocessor.config._transform_unused_feature_types = [
            get_feature("Abundance")
        ]
        self.preprocessor._input_columns = [None, None]
        with self.assertRaises(AssertionError) as context:
            self.preprocessor._validate_transform_params(2)
        self.assertIn(
            str(context.exception),
            "`_transform_unused_feature_types` is defined in "
            "MockProcessorConfig but does not match the arity of "
            "the transform function in MockPreprocessor (i.e. len("
            "self.config._transform_unused_feature_types) != "
            "len(self._input_columns) -> "
            "1 != 2).\n"
            "This can be corrected by doing, for example:\n"
            "_transform_unused_feature_types = field(\n"
            "    default_factory=lambda: [None, None], init=False, "
            "repr=False\n"
            ")",
        )

    def test_fit_missing_input_cols(self):
        with self.assertRaises(AssertionError) as context:
            self.preprocessor._fit(self.X, self.y, funcs=[self.preprocessor._fit_arrow])

        self.assertIn(
            "`extra_indices` was returned as `None` from "
            "`MockPreprocessor`. "
            "Was `MockPreprocessor._input_columns` or "
            "`MockPreprocessor.config._input_columns` set correctly?",
            str(context.exception),
        )

    def test_fit_transform(self):
        self.preprocessor.fit_transform(self.X, self.y)

    def test_fit_transform_with_columns(self):
        self.preprocessor.fit_transform(
            self.data, input_columns=self.column_names, target_column=self.target_column
        )

    def test_fit_transform_without_automatic_column_selection(self):
        # First, fit the processor
        with self.assertRaises(AssertionError) as context:
            self.preprocessor.config._fit_input_feature_types = [None, None]
            self.preprocessor.fit_transform(self.data)
        self.assertIn(
            str(context.exception),
            "`MockPreprocessor.fit` requires 2 arguments ('X', 'y'), but only 1 was "
            "provided. Either provide the missing arguments or provide the input "
            "columns found in 'X', if applicable.",
        )

    def test_fit_transform_with_missing_col_in_X(self):
        # First, fit the processor
        with self.assertRaises(AssertionError) as context:
            self.preprocessor.fit_transform(
                self.X
            )  # does not any contain TARGET_FEATURE_TYPES
        self.assertIn(
            str(context.exception),
            "`MockPreprocessor.fit` requires 2 arguments ('X', 'y'), but only 1 was "
            "provided. Either provide the missing arguments or provide the input "
            "columns found in 'X', if applicable.",
        )

    def test_fit_transform_output_format(self):
        # First, fit the processor
        self.preprocessor = self.preprocessor.fit(self.X, self.y)
        out1 = self.preprocessor.transform(self.X, output_format="numpy")
        out2 = self.preprocessor.fit_transform(self.X, self.y, output_format="numpy")
        self.assertIs(type(out1), np.ndarray)
        self.assertTrue(np.array_equal(out1, out2))

    def test_process_extra_inds(self):
        extra_indices = [[1, 2]]
        unused_extra_indices = [[3]]
        extra_inputs = [None]
        orig_input = None
        result = self.preprocessor._process_extra_inds(
            orig_input, extra_inputs, extra_indices, unused_extra_indices
        )
        self.assertEqual(result, (extra_indices, unused_extra_indices))

    def test_prepare_fit_kwargs(self):
        combined_inputs = self.data
        orig_input = self.sample_metadata
        extra_inputs = [self.X, self.y]
        selected_indices = list(range(self.data.shape[1]))
        extra_indices = [
            list(range(self.X.shape[1])),
            list(range(self.y.shape[1])),
        ]
        map_kwargs, pooler = self.preprocessor._prepare_fit_kwargs(
            funcs=[
                self.preprocessor._fit_arrow,
                self.preprocessor._fit_pandas,
                self.preprocessor._fit_numpy,
            ],
            combined_inputs=combined_inputs,
            orig_input=orig_input,
            selected_indices=selected_indices,
            extra_inputs=extra_inputs,
            extra_indices=extra_indices,
            extra_untouched_inputs=None,
            map_kwargs={"fn_kwargs": {}},
            num_proc=1,
        )

        # The expected extra indices should be adjusted to match the index
        # within the combined_inputs (i.e. offset by the number of columns
        # in the orig_input)
        expected_extra_indices = [
            list(
                range(
                    self.sample_metadata.shape[1],
                    self.sample_metadata.shape[1] + self.X.shape[1],
                )
            ),
            list(
                range(
                    self.sample_metadata.shape[1] + self.X.shape[1],
                    self.data.shape[1],
                )
            ),
        ]
        expected_map_kwargs = {
            "fn_kwargs": {
                "fn": self.preprocessor._fit_pandas,  # should match data type of combined_inputs
                "func_type": "_fit",
                "extra_untouched_inputs": None,
                "selected_indices": selected_indices,
                "unused_indices": None,
                "extra_indices": expected_extra_indices,  # should be adjusted to match the data type of extra_inputs
                "unused_extra_indices": [None, None],
                "with_metadata": False,
                "in_format_kwargs": {"target_format": "pandas"},
                "out_format_kwargs": {"target_format": None},
            },
            "with_indices": False,
            "with_rank": False,
            "desc": "Fitting test",
            "batched": True,
            "batch_size": None,
            "new_fingerprint": "None-processor-mock",
        }
        for key, value in expected_map_kwargs.items():
            if key == "fn_kwargs":
                for k, v in value.items():
                    self.assertEqual(
                        map_kwargs[key][k], v, f"{k}: {map_kwargs[key][k]} != {v}"
                    )
            else:
                self.assertEqual(map_kwargs[key], value)
        self.assertIsNone(pooler)

    def test_prepare_transform_kwargs(self):
        combined_inputs = self.data
        orig_input = self.sample_metadata
        selected_indices = list(range(self.data.shape[1]))
        extra_indices = [
            list(range(self.X.shape[1])),
            list(range(self.y.shape[1])),
        ]
        map_kwargs = self.preprocessor._prepare_transform_kwargs(
            combined_inputs,
            orig_input,
            self.X,
            self.y,
            selected_indices=selected_indices,
            extra_indices=extra_indices,
            unused_indices=None,
            unused_extra_indices=None,
            map_kwargs={"fn_kwargs": {}},
            num_proc=1,
        )

        # The expected extra indices should be adjusted to match the index
        # within the combined_inputs (i.e. offset by the number of columns
        # in the orig_input)
        expected_extra_indices = [
            list(
                range(
                    self.sample_metadata.shape[1],
                    self.sample_metadata.shape[1] + self.X.shape[1],
                )
            ),
            list(
                range(
                    self.sample_metadata.shape[1] + self.X.shape[1],
                    self.data.shape[1],
                )
            ),
        ]
        expected_map_kwargs = {
            "fn_kwargs": {
                "fn": self.preprocessor._transform_pandas,  # should match data type of combined_inputs
                "func_type": "_transform",
                "with_metadata": False,
                "selected_indices": selected_indices,
                "unused_indices": None,
                "extra_indices": expected_extra_indices,
                "unused_extra_indices": [None, None],
                "keep_unused_columns": None,
                "in_format_kwargs": {"target_format": "pandas"},
                "out_format_kwargs": {"target_format": "arrow"},
                "feature_names": list(self.sample_metadata.columns),
            },
            "with_indices": False,
            "with_rank": False,
            "desc": "Transforming test",
            "keep_in_memory": False,
            "cache_file_name": None,
            "num_proc": 1,
            "batched": True,
            "batch_size": 1000,
            "load_from_cache_file": True,
            "new_fingerprint": None,
        }

        for key, value in expected_map_kwargs.items():
            if key == "fn_kwargs":
                for k, v in value.items():
                    self.assertEqual(
                        map_kwargs[key][k], v, f"{k}: {map_kwargs[key][k]} != {v}"
                    )
            else:
                self.assertEqual(map_kwargs[key], value)

    def test_process_transform_input(self):
        X = "input_data"
        kwargs = {"key": "value"}
        result_X, result_kwargs = self.preprocessor._process_transform_input(
            X, **kwargs
        )
        self.assertEqual(result_X, X)
        self.assertEqual(result_kwargs, kwargs)

    def test_process_transform_output(self):
        result = self.preprocessor._process_transform_output(self.X, self.X)
        pd.testing.assert_frame_equal(result, self.X)

    def test_get_params(self):
        params = self.preprocessor.get_params()
        self.assertIsInstance(params, dict)
        self.assertEqual(params["processor_param_int"], 1)

    def test_load_processed_estimator_from_cache(self):
        with self.assertRaises(Exception):
            self.preprocessor.load_processed_estimator_from_cache(
                "nonexistent_cache_file"
            )

    # TODO: Make a better test for this
    # def test_get_features_out(self):
    #     X = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    #     features_out = self.preprocessor._get_features_out(X, selected_indices=[0])
    #     self.assertIsInstance(features_out, dict)

    def test_prepare_runner(self):
        fn_kwargs = {}
        result_kwargs = self.preprocessor._prepare_runner(None, **fn_kwargs)
        self.assertEqual(result_kwargs, fn_kwargs)

    # def test_run(self):
    #     def runner(X, **kwargs):
    #         return "processed"
    #
    #     result = self.preprocessor.run("input_data", runner=runner)
    #     self.assertEqual(result, "processed")

    def test_get_feature_names_out(self):
        # Testing _get_feature_names_out method
        input_features = ["feature1", "feature2"]
        n_features_out = 2
        useful_feature_inds = [0, 1]
        out_features = self.preprocessor._get_feature_names_out(
            input_features=input_features,
            n_features_out=n_features_out,
            useful_feature_inds=useful_feature_inds,
            one_to_one_features=True,
        )
        self.assertEqual(out_features, ["feature1", "feature2"])

        # Test with prefix and suffix
        self.preprocessor.config.features_out_prefix = "prefix_"
        self.preprocessor.config.features_out_suffix = "_suffix"
        out_features = self.preprocessor._get_feature_names_out(
            input_features=input_features,
            n_features_out=n_features_out,
            useful_feature_inds=useful_feature_inds,
            one_to_one_features=True,
        )
        self.assertEqual(
            out_features, ["prefix_feature1_suffix", "prefix_feature2_suffix"]
        )

    # def test_transform_with_unused_columns(self):
    #     # test transforming data while keeping unused columns
    #     x = self.x.copy()
    #     self.preprocessor.fit(x[["feature1", "feature2"]])
    #
    #     transformed_x = self.preprocessor.transform(x, keep_unused_columns=true)
    #
    #     # verify that unused columns are present
    #     self.assertin("metadata", transformed_x.columns)
    #     self.assertin("target", transformed_x.columns)
    #     # verify that 'feature1' and 'feature2' are centered
    #     expected_x = x[["feature1", "feature2"]] - x[["feature1", "feature2"]].mean()
    #     pd.testing.assert_frame_equal(
    #         transformed_x[["feature1", "feature2"]], expected_x
    #     )
    #     # verify that unused columns are unchanged
    #     pd.testing.assert_series_equal(transformed_x["metadata"], x["metadata"])
    #     pd.testing.assert_series_equal(transformed_x["target"], x["target"])

    def test_process_fit_batch_input_valid(self):
        """
        Test _process_fit_batch_input with valid arguments.
        """
        input_processed, fn_args, fn_kwargs = self.model._process_fit_batch_input(
            DataHandler.to_arrow(self.data),
            fn=self.model._fit_sklearn,
            selected_indices=[0, 1, 2],
            extra_indices=[[-1]],
            unused_indices=[],
            extra_untouched_inputs=None,
            with_metadata=False,
            in_format_kwargs={"target_format": "pandas"},
        )

        # Assertions
        self.assertIsNotNone(input_processed)
        self.assertIsInstance(fn_args, tuple)
        self.assertEqual(len(fn_args), 1)
        format = DataHandler.get_format(input_processed)
        self.assertEqual(format, "pandas")

    def test_process_fit_batch_unmatched_arity(self):
        """
        Test _process_fit_batch_input with empty input.
        """
        # pretty much the same as test_fit_transform_without_automatic_column_selection
        # but directly testing the method
        with self.assertRaises(AssertionError) as context:
            self.model._process_fit_batch_input(
                DataHandler.to_arrow(self.data),
                fn=self.model._fit_sklearn,
                selected_indices=[0, 1, 2],
                extra_indices=[],
                unused_indices=[],
                extra_untouched_inputs=None,
                with_metadata=False,
                in_format_kwargs={"target_format": "pandas"},
            )
        self.assertIn(
            str(context.exception),
            "`MockModel.fit` requires 2 arguments ('X', 'y'), but only 1 was "
            "provided. Either provide the missing arguments or provide the input "
            "columns found in 'X', if applicable.",
        )

    def test_process_fit_batch_output_invalid(self):
        """
        Test _process_fit_batch_output with invalid output.
        """
        # This test doesn't really do anything, but it's here for completeness
        output = None

        processed_output = self.model._process_fit_batch_output(output)

        # Assertions
        self.assertIsNone(processed_output)

    def test_process_transform_batch_input_valid(self):
        """
        Test _process_transform_batch_input with valid arguments.
        """
        input_data = self.X

        input_processed, fn_args, _ = self.preprocessor._process_transform_batch_input(
            input_data,
            fn=self.preprocessor._transform_pandas,
            selected_indices=[0, 1, 2],
            unused_indices=[3],
            keep_unused_columns=True,
            with_metadata=False,
            in_format_kwargs={},
            out_format_kwargs={},
        )

        # Assertions
        self.assertIsNotNone(input_processed)
        self.assertIsInstance(fn_args, tuple)
        self.assertEqual(len(fn_args), 0)

    def test_process_transform_batch_input_empty(self):
        """
        Test _process_transform_batch_input with empty input.
        """
        with self.assertRaises(AssertionError) as context:
            input_data = None

            self.preprocessor._process_transform_batch_input(
                input_data,
                fn=self.preprocessor._transform_pandas,
                selected_indices=[0, 1, 2],
                unused_indices=[3],
                keep_unused_columns=True,
                with_metadata=False,
                in_format_kwargs={},
                out_format_kwargs={},
            )

        self.assertIn(
            str(context.exception),
            "No input data was provided for processing.",
        )

    def test_process_transform_batch_input_invalid_columns(self):
        """
        Test _process_transform_batch_input with invalid/missing columns.
        """
        with self.assertRaises(IndexError):
            input_data = self.X

            self.preprocessor._process_transform_batch_input(
                input_data,
                fn=self.preprocessor._transform_pandas,
                selected_indices=[self.X.shape[1] + 1],
                unused_indices=[3],
                keep_unused_columns=True,
                with_metadata=False,
                in_format_kwargs={},
                out_format_kwargs={},
            )

    @require_polars
    def test_process_transform_batch_output_valid(self):
        """
        Test _process_transform_batch_output with valid output.
        """
        input = DataHandler.to_polars(self.X, [0, 1, 2, 3])
        output = DataHandler.to_polars(self.X, [0, 1, 3])

        processed_output = self.preprocessor._process_transform_batch_output(
            input,
            output,
            selected_indices=[0, 1, 3],
            unused_indices=[2],
            keep_unused_columns=True,
            feature_names=["test1", "test2", "int32_3", "test4"],
            out_format_kwargs={"target_format": "arrow"},
            one_to_one_features=True,
        )

        # Assertions
        self.assertEqual(
            processed_output.column_names, ["test1", "test2", "int32_3", "test4"]
        )
        self.assertIsInstance(processed_output, pa.Table)

    # def test_process_transform_batch_output_keep_unused_columns(self):
    #     """
    #     Test _process_transform_batch_output with keeping unused columns.
    #     """
    #     transformed_X = self.preprocessor._transform_pandas(self.X)
    #
    #     processed_output = self.preprocessor._process_transform_batch_output(
    #         self.X,
    #         transformed_X,
    #         fn_kwargs={
    #             "selected_indices": [0, 1, 2],
    #             "unused_indices": [3],
    #             "keep_unused_columns": True,
    #             "feature_names": ["sample_id", "multi_int", "multi_str", "labels"],
    #             "out_format_kwargs": {"target_format": "pandas"},
    #         },
    #     )
    #
    #     # Assertions
    #     self.assertIn("labels", processed_output.column_names)
    #     self.assertEqual(len(processed_output.columns), 4)

    # def test_process_transform_batch_output_discard_unused_columns(self):
    #     """
    #     Test _process_transform_batch_output with discarding unused columns.
    #     """
    #     transformed_X = self.preprocessor._transform_pandas(self.X)
    #
    #     processed_output = self.preprocessor._process_transform_batch_output(
    #         self.X,
    #         transformed_X,
    #         fn_kwargs={
    #             "selected_indices": [0, 1, 2],
    #             "unused_indices": [3],
    #             "keep_unused_columns": False,
    #             "feature_names": ["sample_id", "multi_int", "multi_str"],
    #             "out_format_kwargs": {"target_format": "pandas"},
    #         },
    #     )
    #
    #     # Assertions
    #     self.assertNotIn("labels", processed_output.columns)
    #     self.assertEqual(len(processed_output.columns), 3)

    # def test_process_transform_batch_output_invalid_output_format(self):
    #     """
    #     Test _process_transform_batch_output with invalid output format.
    #     """
    #     transformed_X = self.preprocessor._transform_pandas(self.X)
    #
    #     with self.assertRaises(ValueError):
    #         self.preprocessor._process_transform_batch_output(
    #             self.X,
    #             transformed_X,
    #             fn_kwargs={
    #                 "selected_indices": [0, 1, 2],
    #                 "unused_indices": [3],
    #                 "keep_unused_columns": False,
    #                 "feature_names": ["sample_id", "multi_int", "multi_str"],
    #                 "out_format_kwargs": {"target_format": "invalid_format"},
    #             },
    #         )

    # def test_process_fit_batch_input_invalid_fn_kwargs(self):
    #     """
    #     Test _process_fit_batch_input with invalid fn_kwargs.
    #     """
    #     with self.assertRaises(AttributeError):
    #         input_data = self.X
    #         y = self.y
    #
    #         # Pass incorrect fn_kwargs (missing 'fn')
    #         self.model._process_fit_batch_input(
    #             input_data,
    #             y,
    #             fn_kwargs={
    #                 "selected_indices": [0, 1, 2],
    #                 "extra_indices": [],
    #                 "unused_indices": [],
    #                 "extra_untouched_inputs": [],
    #                 "with_metadata": False,
    #                 "indicate_last_batch": False,
    #                 "in_format_kwargs": {},
    #                 "out_format_kwargs": {},
    #                 "with_target": True,
    #             },
    #         )

    # def test_process_transform_batch_input_with_extra_inputs(self):
    #     """
    #     Test _process_transform_batch_input with extra inputs.
    #     """
    #     with patch.object(
    #         self.preprocessor, "_transform_pandas", return_value=self.X
    #     ) as mock_transform:
    #         input_data = self.X
    #         extra_input = pa.table({"extra_col": [4, 5, 6]})
    #
    #         input_processed, fn_args, fn_kwargs = (
    #             self.preprocessor._process_transform_batch_input(
    #                 input_data,
    #                 extra_input,
    #                 fn_kwargs={
    #                     "fn": self.preprocessor._transform_pandas,
    #                     "selected_indices": [0, 1, 2],
    #                     "unused_indices": [3],
    #                     "keep_unused_columns": True,
    #                     "with_metadata": False,
    #                     "in_format_kwargs": {},
    #                     "out_format_kwargs": {},
    #                 },
    #             )
    #         )
    #
    #         # Assertions
    #         self.assertIsNotNone(input_processed)
    #         self.assertIsInstance(fn_args, tuple)
    #         self.assertEqual(len(fn_args), 1)
    #         self.assertIsInstance(fn_args[0], pa.Table)
    #         self.assertEqual(fn_kwargs["fn"], self.preprocessor._transform_pandas)
    #         mock_transform.assert_called_once()

    def test_process_fit_input(self):
        input = "input_data"
        kwargs = {"key": "value"}
        result_input, result_kwargs = self.preprocessor._process_fit_input(
            input, **kwargs
        )
        self.assertEqual(result_input, input)
        self.assertEqual(result_kwargs, kwargs)

    def test_process_fit_output(self):
        input = "input_data"
        out = "output_data"
        result = self.preprocessor._process_fit_output(input, out)
        self.assertEqual(result, out)

    # def test_repr(self):
    #     repr_str = repr(self.preprocessor)
    #     self.assertIsInstance(repr_str, str)
    #     self.assertIn("MockPreprocessor", repr_str)
    #
    # def test_repr_mimebundle(self):
    #     mimebundle = self.preprocessor._repr_mimebundle_()
    #     self.assertIsInstance(mimebundle, dict)
    #     self.assertIn("text/plain", mimebundle)
    #
    # def test_shard(self):
    #     # Testing shard method
    #     num_shards = 5
    #     for index in range(num_shards):
    #         shard = self.preprocessor.shard(
    #             self.X, num_shards=num_shards, index=index, contiguous=True
    #         )
    #         # Verify that the shards cover the entire dataset when combined
    #         if index == 0:
    #             combined_shard = shard
    #         else:
    #             combined_shard = pd.concat([combined_shard, shard], ignore_index=True)
    #     pd.testing.assert_frame_equal(
    #         combined_shard.reset_index(drop=True), self.X.reset_index(drop=True)
    #     )

    # def test_pool_fit(self):
    #     # Testing _pool_fit method
    #     # Since pooling is not implemented for multiple processors, expect NotImplementedError
    #
    #     # Create multiple fitted processors
    #     fitted_processors = [
    #         self.preprocessor.fit(self.X[["feature1", "feature2"]]) for _ in range(2)
    #     ]
    #
    #     with self.assertRaises(NotImplementedError):
    #         self.preprocessor._pool_fit(fitted_processors)
    #
    #     # Test with a single processor
    #     pooled_processor = self.preprocessor._pool_fit([self.preprocessor])
    #     self.assertEqual(pooled_processor, self.preprocessor)
    #
    # def test_map(self):
    #     X = pd.DataFrame({"col1": [1, 2, 3]})
    #
    #     def func(batch):
    #         return batch * 2
    #
    #     result = self.preprocessor.map(X, function=func, batched=True)
    #     expected = X * 2
    #     pd.testing.assert_frame_equal(result[0], expected)
    #
    # def test_map_single(self):
    #     shard = pd.DataFrame({"col1": [1, 2, 3]})
    #
    #     def func(batch):
    #         return batch * 2
    #
    #     gen = BaseProcessor._map_single(shard, function=func, batched=True)
    #     results = list(gen)
    #     for rank, done, content in results:
    #         if done:
    #             processed_data = content
    #     expected = shard * 2
    #     pd.testing.assert_frame_equal(processed_data[0], expected)
