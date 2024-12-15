import unittest

import pandas as pd
from biocore.data_handling import DataHandler
from biocore.utils.import_util import (
    is_biosets_available,
    is_datasets_available,
    is_polars_available,
    is_rpy2_available,
)


def require_polars(test_case):
    """
    Decorator marking a test that requires Polars.

    These tests are skipped when Polars isn't installed.

    """
    if not is_polars_available():
        test_case = unittest.skip("test requires Polars")(test_case)
    return test_case


def require_biosets(test_case):
    """
    Decorator marking a test that requires biosets.

    These tests are skipped when biosets isn't installed.

    """
    if not is_biosets_available():
        test_case = unittest.skip("test requires biosets")(test_case)
    return test_case


def require_matplotlib(test_case):
    """
    Decorator marking a test that requires matplotlib.

    These tests are skipped when matplotlib isn't installed.

    """
    try:
        import matplotlib  # noqa
    except ImportError:
        test_case = unittest.skip("test requires matplotlib")(test_case)
    return test_case


def require_datasets(test_case):
    """
    Decorator marking a test that requires datasets.

    These tests are skipped when datasets isn't installed.

    """
    if not is_datasets_available():
        test_case = unittest.skip("test requires datasets")(test_case)
    return test_case


def require_rpy2(test_case):
    """
    Decorator marking a test that requires rpy2.

    These tests are skipped when rpy2 isn't installed.

    """
    if not is_rpy2_available():
        test_case = unittest.skip("test requires rpy2")(test_case)
    return test_case


def create_bioset(
    X,
    y=None,
    sample_metadata=None,
    with_feature_metadata=True,
    feature_type=None,
    target_type=None,
):
    """
    Create a bioset from data.

    Args:
        X (np.ndarray or pd.DataFrame): The data matrix.
        y (np.ndarray or pd.Series, optional): The target vector.
        metadata (pd.DataFrame, optional): The metadata.
        feature_metadata (pd.DataFrame, optional): The feature metadata.

    Returns:
        bioset (Bioset): The bioset.

    """
    from biosets.features import Metadata, Sample, Batch
    from datasets import ClassLabel, Features

    data = [X]
    if y is not None:
        data.append(y)
    if sample_metadata is not None:
        data = [sample_metadata] + data

    data = pd.concat(data, axis=1)
    dtypes = DataHandler.get_dtypes(X)

    features = {}
    if sample_metadata is not None:
        sample_metadata_dtypes = DataHandler.get_dtypes(sample_metadata)
        for k, v in sample_metadata_dtypes.items():
            if "sample" in k.lower():
                features[k] = Sample(v)
            elif "batch" in k.lower():
                features[k] = Batch(v)
            else:
                features[k] = Metadata(v)

    if with_feature_metadata:
        metadata = {
            "my_metadata_str": "str",
            "my_metadata_int": 0,
            "my_metadata_float": 0.0,
            "my_metadata_bool": False,
            "my_metadata_list": ["a", "b", "c"],
            "my_metadata_dict": {"a": 1, "b": 2, "c": 3},
            "my_metadata_none": None,
        }
        features.update(
            {k: feature_type(dtype=v, metadata=metadata) for k, v in dtypes.items()}
        )
    else:
        features.update({k: feature_type(dtype=v) for k, v in dtypes.items()})
    if y is not None and target_type is not None:
        if issubclass(target_type, ClassLabel):
            names = list(set(y.values.flatten().tolist()))

            if isinstance(names[0], int):
                names = ["abcdefghiklmnopqrstuvwxyz"[i] for i in names if i >= 0]

            features[y.columns[0]] = target_type(num_classes=len(names), names=names)
        else:
            features[y.columns[0]] = target_type()

    ds = DataHandler.to_bioset(data)
    ds._info.features = Features(features)
    return ds
