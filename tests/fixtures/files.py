import contextlib
import csv
import json
import os
import sqlite3
import tarfile
import textwrap
import zipfile
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from biocore.utils.import_util import (
    is_biosets_available,
    requires_backends,
)
from biofit.integration.biosets import get_feature
from biofit.utils import enable_full_determinism
from biofit.utils.py_util import set_seed
from sklearn.datasets import make_classification

# Constants
ALPHANUMERIC = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
SEED = 42
enable_full_determinism(SEED)
PA_DATA = {
    "null": lambda num_rows: np.array([None] * num_rows),
    "bool": lambda num_rows: np.random.choice([True, False], size=num_rows),
    "one_hot": lambda num_rows: np.random.choice([0, 1], size=num_rows),
    "multi_bins": lambda num_rows: np.random.choice([0, 1], size=num_rows),
    "int8": lambda num_rows: np.random.normal(loc=0, scale=127, size=num_rows).astype(
        np.int8
    ),
    "int16": lambda num_rows: np.random.normal(
        loc=0, scale=32767, size=num_rows
    ).astype(np.int16),
    "int32": lambda num_rows: np.random.normal(
        loc=0, scale=2147483647, size=num_rows
    ).astype(np.int32),
    "int64": lambda num_rows: np.random.normal(
        loc=0, scale=9223372036854775807, size=num_rows
    ).astype(np.int64),
    "uint8": lambda num_rows: np.abs(
        np.random.normal(loc=127.5, scale=127.5, size=num_rows)
    ).astype(np.uint8),
    "uint16": lambda num_rows: np.abs(
        np.random.normal(loc=32767.5, scale=32767.5, size=num_rows)
    ).astype(np.uint16),
    "uint32": lambda num_rows: np.abs(
        np.random.normal(loc=2147483647.5, scale=2147483647.5, size=num_rows)
    ).astype(np.uint32),
    "uint64": lambda num_rows: np.abs(
        np.random.normal(
            loc=9223372036854775807.5, scale=9223372036854775807.5, size=num_rows
        )
    ).astype(np.uint64),
    "float16": lambda num_rows: np.random.normal(size=num_rows).astype(np.float16),
    "float32": lambda num_rows: np.random.normal(size=num_rows).astype(np.float32),
    "float64": lambda num_rows: np.random.normal(size=num_rows),
    "string": lambda num_rows: np.array(
        [
            "".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), size=5))
            for _ in range(num_rows)
        ]
    ),
    "binary": lambda num_rows: np.array(
        [
            bytes(
                "".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), size=5)),
                "utf-8",
            )
            for _ in range(num_rows)
        ]
    ),
    "large_string": lambda num_rows: np.array(
        [
            "".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), size=5))
            for _ in range(num_rows)
        ]
    ),
    "large_binary": lambda num_rows: np.array(
        [
            bytes(
                "".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), size=5)),
                "utf-8",
            )
            for _ in range(num_rows)
        ]
    ),
    "date32": lambda num_rows: np.array(
        [np.random.randint(1, 2147483647, size=num_rows).tolist()]
    ),
    "date64": lambda num_rows: np.array(
        [np.random.randint(1, 2147483647, size=num_rows).tolist()]
    ),
    "time32": lambda num_rows: np.random.normal(
        loc=0, scale=2147483647, size=num_rows
    ).astype(np.int32),
    "time64": lambda num_rows: np.random.normal(
        loc=0, scale=9223372036854775807, size=num_rows
    ).astype(np.int64),
    "timestamp": lambda num_rows: np.array(
        [
            pd.Timestamp(np.random.choice(pd.date_range("2020-01-01", periods=365)))
            for _ in range(num_rows)
        ]
    ),
    "duration": lambda num_rows: np.random.normal(
        loc=500, scale=250, size=num_rows
    ).astype(int),
    "decimal128": lambda num_rows: np.array(
        [Decimal(np.random.randint(1, 1000)) for _ in range(num_rows)]
    ),
    "struct": lambda num_rows: [
        {"a": np.random.randint(1, 1000)} for _ in range(num_rows)
    ],
}


PA_FIELDS = {
    "null": pa.field("null", pa.null()),
    "bool": pa.field("bool", pa.bool_()),
    "int8": pa.field("int8", pa.int8()),
    "int16": pa.field("int16", pa.int16()),
    "int32": pa.field("int32", pa.int32()),
    "int64": pa.field("int64", pa.int64()),
    "uint8": pa.field("uint8", pa.uint8()),
    "uint16": pa.field("uint16", pa.uint16()),
    "uint32": pa.field("uint32", pa.uint32()),
    "uint64": pa.field("uint64", pa.uint64()),
    "float16": pa.field("float16", pa.float16()),
    "float32": pa.field("float32", pa.float32()),
    "float64": pa.field("float64", pa.float64()),
    "string": pa.field("string", pa.string()),
    "binary": pa.field("binary", pa.binary()),
    "large_string": pa.field("large_string", pa.large_string()),
    "large_binary": pa.field("large_binary", pa.large_binary()),
    "date32": pa.field("date32", pa.date32()),
    "date64": pa.field("date64", pa.date64()),
    "time32": pa.field("time32", pa.time32("s")),
    "time64": pa.field("time64", pa.time64("ns")),
    "timestamp": pa.field("timestamp", pa.timestamp("s")),
    "duration": pa.field("duration", pa.duration("s")),
    "decimal128": pa.field("decimal128", pa.decimal128(38, 9)),
    "struct": pa.field("struct", pa.struct([pa.field("a", pa.int32())])),
}


def create_directory(path):
    """
    Create a directory if it doesn't exist.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {path}: {e}")
        raise


def _create_all_arrow_types_dataframe(num_rows=100, feature_type=None):
    from datasets.features import Value

    if feature_type is None:
        feature_type = Value
    data = {k: v(num_rows) for k, v in PA_DATA.items()}

    features = {
        k: feature_type(k) if k != "struct" else {"a": feature_type("int32")}
        for k in PA_FIELDS.keys()
    }

    return data, features


def create_omic_dataset(
    num_rows=100,
    num_cols=None,
    dtype="all",
    sample="sample_id",
    batch="batch",
    label="label",
    multi_class=False,
    task="classification",
    label_type="int",
    metadata=True,
    input_feature=None,
    sparse=False,
    missing_labels=False,
):
    """
    Create a sample dataframe with predefined structure.
    """
    from datasets import Value

    if input_feature is None:
        input_feature = Value
    data = {}
    features = {}
    enable_full_determinism(SEED)

    if sample:
        data[sample] = [str(i) for i in range(num_rows)]
        features[sample] = get_feature("Sample")("string")
    if batch:
        data[batch] = [str(i) for i in range(num_rows)]
        features[batch] = get_feature("Batch")("string")
    metadata_value_options = {
        "multi_classification_int": [i % 3 for i in range(num_rows)],
        "multi_classification_str": [
            ALPHANUMERIC[i % len(ALPHANUMERIC)] for i in range(num_rows)
        ],
        "bin_classification_bool": [
            True if i % 2 == 0 else False for i in range(num_rows)
        ],
        "bin_classification_int": [i % 2 for i in range(num_rows)],
        "bin_classification_str": [
            "positive" if i > num_rows // 2 else "negative" for i in range(num_rows)
        ],
        "regression": np.random.randn(num_rows),
    }
    metadata_feature_options = {
        "multi_classification_int": "int8",
        "multi_classification_str": "string",
        "bin_classification_bool": "bool",
        "bin_classification_int": "int8",
        "bin_classification_str": "string",
        "regression": "float32",
    }
    if label:
        if task == "classification":
            if multi_class:
                label_name = "multi"
            else:
                label_name = "bin"
            label_name += f"_classification_{label_type}"
        else:
            label_name = "regression"

        data[label] = metadata_value_options.pop(label_name)
        if missing_labels:
            # Randomly set 10% of labels to -1 if classification, else set to None
            if task == "classification":
                indices_to_replace = np.random.choice(
                    num_rows, int(num_rows * 0.1), replace=False
                )
                data[label] = [
                    -1 if i in indices_to_replace else lab
                    for i, lab in enumerate(data[label])
                ]
            else:
                indices_to_replace = np.random.choice(
                    num_rows, int(num_rows * 0.1), replace=False
                )
                data[label] = [
                    None if i in indices_to_replace else lab
                    for i, lab in enumerate(data[label])
                ]
        label_dtype = metadata_feature_options.pop(label_name)
        if label_name == "regression":
            features[label] = get_feature("RegressionTarget")(label_dtype)
        else:
            names = list(set(data[label]))
            if not isinstance(names[0], str):
                names = [str(n) for n in names]
            else:
                name_map = {n: i for i, n in enumerate(names)}
                data[label] = [name_map[n] for n in data[label]]

            num_classes = len(names)

            features[label] = get_feature("ClassLabel")(
                num_classes=num_classes, names=names
            )
    if metadata:
        if isinstance(metadata, str):
            data.update(metadata_value_options)
            features.update(
                {
                    k: get_feature("Metadata")(dtype=v)
                    for k, v in metadata_feature_options.items()
                }
            )
        else:
            for label, v in metadata_value_options.items():
                data[label] = v
                features[label] = Value(metadata_feature_options[label])

    if dtype == "all":
        ext_data, ext_features = _create_all_arrow_types_dataframe(
            num_rows=num_rows, feature_type=input_feature
        )
    else:
        ext_data = {}
        ext_features = {}
        if sparse and isinstance(sparse, bool):
            sparse = 0.8
        if num_cols is None:
            num_cols = 1

        dtype_to_pa = {
            "multi_bins": "int32",
            "one_hot": "int32",
        }

        for i in range(num_cols):
            arr: np.ndarray = PA_DATA[dtype](num_rows)
            ext_data[f"{dtype}_{i}"] = arr.tolist()

            ext_features[f"{dtype}_{i}"] = input_feature(
                dtype=dtype_to_pa.get(dtype, dtype),
                metadata={
                    "my_metadata_str": ALPHANUMERIC[
                        np.random.randint(0, len(ALPHANUMERIC))
                    ],
                    "my_metadata_int": np.random.randint(0, 100),
                },
            )
        if sparse:
            mat = np.array([ext_data[f"{dtype}_{i}"] for i in range(num_cols)]).T
            for i in range(num_cols):
                arr = np.array(ext_data[f"{dtype}_{i}"])
                total_values = arr.size
                if isinstance(sparse, list):
                    _sparse = sparse[i]
                else:
                    _sparse = sparse
                if isinstance(_sparse, bool):
                    _sparse = np.random.uniform(0.1, 0.9)

                num_to_replace = max(
                    min(int(total_values * (_sparse)), total_values - 1), 0
                )
                indices_to_replace = np.random.choice(
                    total_values, num_to_replace, replace=False
                )
                # check if replacing with 0 would make a row all 0s
                for idx in indices_to_replace:
                    if dtype in [
                        "one_hot",
                        "multi_bins",
                        "uint8",
                        "uint16",
                        "uint32",
                        "uint64",
                    ]:
                        if np.sum(mat[:idx] > 0) + np.sum(mat[idx + 1 :] > 0) == 0:
                            indices_to_replace = np.delete(
                                indices_to_replace, np.where(indices_to_replace == idx)
                            )
                        else:
                            arr[idx] = 0
                    else:
                        if all(v is None for v in mat[:idx]) and all(
                            v is None for v in mat[idx + 1 :]
                        ):
                            indices_to_replace = np.delete(
                                indices_to_replace, np.where(indices_to_replace == idx)
                            )
                        else:
                            arr[idx] = 0
                ext_data[f"{dtype}_{i}"] = arr.tolist()

    data.update(ext_data)
    features.update(ext_features)
    if is_biosets_available():
        import biosets
        import datasets

        return biosets.Bioset.from_dict(data, features=datasets.Features(features))
    return pd.DataFrame(data)


def create_feature_dataframe(num_cols=100, feature_id="feature"):
    """
    Create a feature dataframe with predefined structure.
    """
    enable_full_determinism(SEED)
    data = {
        feature_id: [str(i) for i in range(num_cols)],
    }

    fields = [
        pa.field(feature_id, pa.string()),
    ]

    ext_data, ext_fields = _create_all_arrow_types_dataframe(num_rows=num_cols)
    data.update(ext_data)
    fields.extend(ext_fields)

    return pa.table(data, schema=pa.schema(fields))


def directory_exists_with_files(path, expected_files):
    """
    Check if a directory exists with the expected files.
    """
    if not os.path.exists(path):
        return False
    if not all(os.path.exists(os.path.join(path, file)) for file in expected_files):
        return False
    return True


# def save_dataframes(dfs, data_dir, filenames):
#     """
#     Save a list of dataframes to CSV in the specified directory.
#     """
#     for df, filename in zip(dfs, filenames):
#         file_ext = filename.split(".")[-1]
#         if file_ext in ["parquet"]:
#             tbl = pa.Table.from_pandas(df) if isinstance(df, pd.DataFrame) else df
#             if "float16" in tbl.schema.names:
#                 tbl = tbl.drop(["float16"])  # not supported by parquet
#             writer = ParquetWriter(
#                 path=os.path.join(data_dir, filename), schema=tbl.schema
#             )
#             writer.write_table(tbl)
#         elif file_ext in ["arrow"]:
#             tbl = pa.Table.from_pandas(df) if isinstance(df, pd.DataFrame) else df
#             writer = ArrowWriter(
#                 path=os.path.join(data_dir, filename), schema=tbl.schema
#             )
#             writer.write_table(tbl)
#         elif file_ext in ["csv"]:
#             df.to_csv(os.path.join(data_dir, filename), index=False)
#         elif file_ext in ["tsv", "txt"]:
#             df.to_csv(os.path.join(data_dir, filename), sep="\t", index=False)


# def create_fake_data_dir(data, base_dir, overwrite=False):
#     for name, filenames, dfs, _ in data:
#         data_dir = f"{base_dir}/{name}"
#         os.makedirs(data_dir, exist_ok=True)
#         if not directory_exists_with_files(data_dir, filenames) or overwrite:
#             save_dataframes(dfs, data_dir, filenames)


def create_dataset_with_sklearn(
    path,
    experiment_type,
    n_features=20,
    n_samples=50,
    multi_class=False,
    lab_as_str=True,
    dataset_type="snp",
    label_column="label",
    add_missing_labels=False,
):
    if multi_class:
        num_classes = 3
    else:
        num_classes = 2

    samples, labs, names = create_data(
        n_samples,
        n_features,
        num_classes,
        experiment_type,
        lab_as_str=lab_as_str,
        _add_missing_labels=add_missing_labels,
    )
    data = create_sample_metadata(n_samples)
    features = None
    if is_biosets_available():
        if experiment_type == "snp":
            features = create_features(
                n_features, get_feature("GenomicVariant"), "int8", num_classes, names
            )
        elif experiment_type in ["otu", "asv"]:
            features = create_features(
                n_features, get_feature("Abundance"), "int32", num_classes, names
            )
        elif experiment_type == "maldi":
            features = create_features(
                n_features, get_feature("PeakIntensity"), "int32", num_classes, names
            )

    data = pd.concat(
        [
            data,
            pd.DataFrame(samples, columns=[f"int32_{i}" for i in range(n_features)]),
            pd.DataFrame(labs.reshape(-1, 1), columns=[label_column]),
        ],
        axis=1,
    )
    if is_biosets_available():
        import biosets
        import datasets

        ds = biosets.Dataset.from_pandas(data, features=datasets.Features(features))

        ds.info.builder_name = dataset_type
        os.makedirs(path, exist_ok=True)
        ds.save_to_disk(path)
    else:
        data.to_csv(path, index=False)


def _add_missing_labels(labs, num_classes=2, lab_as_str=True, n_samples=50):
    if isinstance(labs, np.ndarray):
        new_labs = labs.tolist()
    else:
        new_labs = labs
    # make 10% of labels missing for each class
    for i in range(num_classes):
        if lab_as_str:
            indices_to_replace = np.random.choice(
                np.where(labs == ALPHANUMERIC[i % len(ALPHANUMERIC)])[0],
                int(n_samples * 0.1),
                replace=False,
            )
            for idx in indices_to_replace:
                new_labs[idx] = None
        else:
            indices_to_replace = np.random.choice(
                np.where(labs == i)[0], int(n_samples * 0.1), replace=False
            )
            for idx in indices_to_replace:
                new_labs[idx] = -1
    return np.array(new_labs)


def create_data(
    n_samples,
    n_features,
    num_classes,
    experiment_type,
    lab_as_str=False,
    add_missing_labels=False,
):
    samples, labs = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=num_classes,
        n_informative=int(n_features * 0.8),
        n_redundant=int(n_features * 0.1),
        random_state=SEED,
    )

    if experiment_type in ["snp"]:
        median = np.median(samples.flatten())
        samples[samples < median] = 0
        samples[samples >= median] = 1
        samples = samples.astype(np.int32)
    elif experiment_type in ["otu", "asv", "maldi"]:
        samples = np.abs(np.round(np.quantile(samples.flatten(), 0.25))) + samples
        samples[samples < 0] = 0
        samples = samples.astype(np.int32)
    if lab_as_str:
        labs = np.array(
            [ALPHANUMERIC[i % len(ALPHANUMERIC)] for i in sorted(labs.tolist())]
        )
        names = [ALPHANUMERIC[i % len(ALPHANUMERIC)] for i in range(num_classes)]
    else:
        names = [str(i) for i in range(num_classes)]
        labs = labs.astype(np.int32)

    if add_missing_labels:
        labs = _add_missing_labels(
            labs, num_classes=num_classes, lab_as_str=lab_as_str, n_samples=n_samples
        )

    samples = pd.DataFrame(samples, columns=[f"feature_{i}" for i in range(n_features)])
    return samples, labs, names


def create_features(
    n_features,
    FeatureType=None,
    dtype=None,
    num_classes=None,
    names=None,
    with_metadata=True,
    as_dict=False,
):
    metadata_feature_options = {
        "multi_int": "int32",
        "multi_str": "string",
        "bin_bool": "bool",
        "bin_int": "int32",
        "bin_str": "string",
        "floating": "float64",
    }
    if not as_dict:
        requires_backends("create_features_for_sample_metadata", "biosets")

        features = {}
        if with_metadata:
            features.update(
                {
                    "sample_id": get_feature("Sample")("string"),
                }
            )
            features.update(
                {
                    k: get_feature("Metadata")(dtype=v)
                    for k, v in metadata_feature_options.items()
                }
            )

        for i in range(n_features):
            features[f"feature_{i}"] = FeatureType(
                dtype="int32",
                metadata={
                    "my_metadata_str": ALPHANUMERIC[
                        np.random.randint(0, len(ALPHANUMERIC))
                    ],
                    "my_metadata_int": np.random.randint(0, 100),
                },
            )
        if num_classes is not None:
            if num_classes > 2:
                features["label"] = get_feature("ClassLabel")(
                    num_classes=num_classes, names=names
                )
            else:
                features["label"] = get_feature("BinClassLabel")(
                    num_classes=num_classes, names=names
                )

    else:
        features = {}
        if with_metadata:
            features = {
                "sample_id": {},
            }
            features.update({k: {} for k in metadata_feature_options.keys()})

        for i in range(n_features):
            features[f"feature_{i}"] = {
                "my_metadata_str": ALPHANUMERIC[
                    np.random.randint(0, len(ALPHANUMERIC))
                ],
                "my_metadata_int": np.random.randint(0, 100),
            }
        if num_classes is not None:
            features["label"] = {
                "num_classes": num_classes,
                "names": names,
            }

    return features


def create_sample_metadata(n_samples=100):
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(n_samples)],
            "multi_int": [i % 3 for i in range(n_samples)],
            "multi_str": [
                ALPHANUMERIC[i % len(ALPHANUMERIC)] for i in range(n_samples)
            ],
            "bin_bool": [True if i % 2 == 0 else False for i in range(n_samples)],
            "bin_int": [i % 2 for i in range(n_samples)],
            "bin_str": [
                "positive" if i > n_samples // 2 else "negative"
                for i in range(n_samples)
            ],
            "floating": np.random.randn(n_samples),
        }
    )


@pytest.fixture(scope="session")
def count_data():
    data, y, _ = create_data(20, 5, 2, "otu")
    y = pd.DataFrame(y[:, None], columns=["label"])
    return data, y


@pytest.fixture(scope="session")
def count_data_multi_class():
    data, y, _ = create_data(20, 5, 3, "otu")
    y = pd.DataFrame(y[:, None], columns=["label"])
    return data, y


@pytest.fixture(scope="session")
def count_data_missing_labels():
    data, y, _ = create_data(20, 5, 2, "otu", add_missing_labels=True)
    y = pd.DataFrame(y[:, None], columns=["label"])
    return data, y


@pytest.fixture(scope="session")
def binary_data():
    data, y, _ = create_data(20, 5, 2, "snp")
    y = pd.DataFrame(y[:, None], columns=["label"])
    return data, y


@pytest.fixture(scope="session")
def float_data():
    data, y, _ = create_data(20, 5, 2, None)
    y = pd.DataFrame(y[:, None], columns=["label"])
    return data, y


@pytest.fixture(scope="session")
def classification_data():
    data, y, _ = create_data(100, 5, 2, None)
    y = pd.DataFrame(y[:, None], columns=["label"])
    return data, y


@pytest.fixture(scope="session")
def classification_data_multi_class():
    data, y, _ = create_data(100, 5, 3, None)
    y = pd.DataFrame(y[:, None], columns=["label"])
    return data, y


@pytest.fixture(scope="session")
def feature_metadata():
    return create_features(5, as_dict=True, with_metadata=False)


@pytest.fixture(scope="session")
def sample_metadata():
    return create_sample_metadata(20)


@pytest.fixture(scope="session")
def biodataset():
    return create_omic_dataset(10, num_cols=3, dtype="float32", metadata="metadata")


@pytest.fixture(scope="session")
def snp_dataset_path(tmp_path_factory):
    set_seed(SEED)
    path = str(tmp_path_factory.mktemp("data") / "SNP")
    create_dataset_with_sklearn(path, "snp")
    return path


@pytest.fixture(scope="session")
def otu_dataset_path(tmp_path_factory):
    set_seed(SEED)
    path = str(tmp_path_factory.mktemp("data") / "OTU")
    create_dataset_with_sklearn(path)
    return path


@pytest.fixture(scope="session")
def otu_dataset_missing_labels_path(tmp_path_factory):
    set_seed(SEED)
    path = str(tmp_path_factory.mktemp("data") / "OTU")
    create_dataset_with_sklearn(path, "otu", lab_as_str=False, add_missing_labels=True)
    return path


@pytest.fixture(scope="session")
def otu_dataset_multi_class_path(tmp_path_factory):
    set_seed(SEED)
    path = str(tmp_path_factory.mktemp("data") / "OTU")
    create_dataset_with_sklearn(path, "otu", multi_class=True)
    return path


@pytest.fixture(scope="session")
def maldi_dataset_path(tmp_path_factory):
    set_seed(SEED)
    path = str(tmp_path_factory.mktemp("data") / "MALDI")
    create_dataset_with_sklearn(path, "maldi")
    return path


@pytest.fixture(scope="session")
def snp_dataset():
    ds = create_omic_dataset(
        num_rows=10,
        num_cols=3,
        dtype="multi_bins",
        metadata="metadata",
        input_feature=get_feature("GenomicVariant"),
        sparse=0.8,
    )
    ds.info.builder_name = "snp"
    return ds


@pytest.fixture(scope="session")
def maldi_dataset():
    ds = create_omic_dataset(
        num_rows=10,
        num_cols=3,
        dtype="multi_bins",
        metadata="metadata",
        input_feature=get_feature("PeakIntensity"),
        sparse=0.8,
    )
    ds.info.builder_name = "maldi"
    return ds


# @pytest.fixture(scope="session")
# def camda_dataset():
#     camda_dir = "./tests/data/CAMDA"
#     camda_metadata_files = os.path.join(camda_dir, "camda.pheno.csv")
#     camda_feature_metadata_files = os.path.join(camda_dir, "camda.feature.csv")
#     ds = load_dataset(
#         "otu",
#         data_dir=camda_dir,
#         sample_metadata_files=camda_metadata_files,
#         feature_metadata_files=camda_feature_metadata_files,
#         label_column="City2",
#         cache_dir="./.cache",
#     )
#     ds.cleanup_cache_files()
#     return ds
#

# @pytest.fixture(scope="session")
# def camda_dataset_files_only():
#     camda_dir = "./tests/data/CAMDA"
#     data_files = os.path.join(camda_dir, "*matrix*.csv")
#     return load_dataset(
#         dataset_type="otu",
#         name="camda",
#         data_files=data_files,
#         label_column="City2",
#     )


# @pytest.fixture(scope="session")
# def camda_dataset_no_polars():
#     camda_dir = "./tests/data/CAMDA"
#     camda_metadata_files = os.path.join(camda_dir, "camda.pheno.csv")
#     camda_feature_metadata_files = os.path.join(camda_dir, "camda.feature.csv")
#     return load_dataset(
#         "otu",
#         data_dir=camda_dir,
#         sample_metadata_files=camda_metadata_files,
#         feature_metadata_files=camda_feature_metadata_files,
#         label_column="City2",
#         cache_dir="./.cache",
#         use_polars=False,
#     )
#

# @pytest.fixture(scope="session")
# def tb_dataset():
#     tb_dir = "./tests/data/genomics_TB"
#     dataset = load_dataset(
#         "snp",
#         "TB",
#         data_dir=tb_dir,
#         label_column="Isoniazid",
#         keep_in_memory=False,
#         cache_dir="./.cache",
#     )
#     dataset.cleanup_cache_files()
#     return dataset
#


@pytest.fixture(scope="session")
def arrow_file(tmp_path_factory, dataset):
    filename = str(tmp_path_factory.mktemp("data") / "file.arrow")
    dataset.map(cache_file_name=filename)
    return filename


# FILE_CONTENT + files


FILE_CONTENT = """\
    Text data.
    Second line of data."""


@pytest.fixture(scope="session")
def text_file(tmp_path_factory):
    filename = tmp_path_factory.mktemp("data") / "file.txt"
    data = FILE_CONTENT
    with open(filename, "w") as f:
        f.write(data)
    return filename


@pytest.fixture(scope="session")
def bz2_file(tmp_path_factory):
    import bz2

    path = tmp_path_factory.mktemp("data") / "file.txt.bz2"
    data = bytes(FILE_CONTENT, "utf-8")
    with bz2.open(path, "wb") as f:
        f.write(data)
    return path


@pytest.fixture(scope="session")
def gz_file(tmp_path_factory):
    import gzip

    path = str(tmp_path_factory.mktemp("data") / "file.txt.gz")
    data = bytes(FILE_CONTENT, "utf-8")
    with gzip.open(path, "wb") as f:
        f.write(data)
    return path


@pytest.fixture(scope="session")
def lz4_file(tmp_path_factory):
    try:
        import lz4.frame

        path = tmp_path_factory.mktemp("data") / "file.txt.lz4"
        data = bytes(FILE_CONTENT, "utf-8")
        with lz4.frame.open(path, "wb") as f:
            f.write(data)
        return path
    except ImportError:
        pytest.skip("lz4 not available")


@pytest.fixture(scope="session")
def seven_zip_file(tmp_path_factory, text_file):
    try:
        import py7zr

        path = tmp_path_factory.mktemp("data") / "file.txt.7z"
        with py7zr.SevenZipFile(path, "w") as archive:
            archive.write(text_file, arcname=os.path.basename(text_file))
        return path
    except ImportError:
        pytest.skip("py7zr not available")


@pytest.fixture(scope="session")
def tar_file(tmp_path_factory, text_file):
    import tarfile

    path = tmp_path_factory.mktemp("data") / "file.txt.tar"
    with tarfile.TarFile(path, "w") as f:
        f.add(text_file, arcname=os.path.basename(text_file))
    return path


@pytest.fixture(scope="session")
def xz_file(tmp_path_factory):
    import lzma

    path = tmp_path_factory.mktemp("data") / "file.txt.xz"
    data = bytes(FILE_CONTENT, "utf-8")
    with lzma.open(path, "wb") as f:
        f.write(data)
    return path


@pytest.fixture(scope="session")
def zip_file(tmp_path_factory, text_file):
    import zipfile

    path = tmp_path_factory.mktemp("data") / "file.txt.zip"
    with zipfile.ZipFile(path, "w") as f:
        f.write(text_file, arcname=os.path.basename(text_file))
    return path


@pytest.fixture(scope="session")
def zstd_file(tmp_path_factory):
    try:
        import zstandard as zstd

        path = tmp_path_factory.mktemp("data") / "file.txt.zst"
        data = bytes(FILE_CONTENT, "utf-8")
        with zstd.open(path, "wb") as f:
            f.write(data)
        return path
    except ImportError:
        pytest.skip("zstandard not available")


# xml_file


@pytest.fixture(scope="session")
def xml_file(tmp_path_factory):
    filename = tmp_path_factory.mktemp("data") / "file.xml"
    data = textwrap.dedent(
        """\
    <?xml version="1.0" encoding="UTF-8" ?>
    <tmx version="1.4">
      <header segtype="sentence" srclang="ca" />
      <body>
        <tu>
          <tuv xml:lang="ca"><seg>Contingut 1</seg></tuv>
          <tuv xml:lang="en"><seg>Content 1</seg></tuv>
        </tu>
        <tu>
          <tuv xml:lang="ca"><seg>Contingut 2</seg></tuv>
          <tuv xml:lang="en"><seg>Content 2</seg></tuv>
        </tu>
        <tu>
          <tuv xml:lang="ca"><seg>Contingut 3</seg></tuv>
          <tuv xml:lang="en"><seg>Content 3</seg></tuv>
        </tu>
        <tu>
          <tuv xml:lang="ca"><seg>Contingut 4</seg></tuv>
          <tuv xml:lang="en"><seg>Content 4</seg></tuv>
        </tu>
        <tu>
          <tuv xml:lang="ca"><seg>Contingut 5</seg></tuv>
          <tuv xml:lang="en"><seg>Content 5</seg></tuv>
        </tu>
      </body>
    </tmx>"""
    )
    with open(filename, "w") as f:
        f.write(data)
    return filename


DATA = [
    {"col_1": "0", "col_2": 0, "col_3": 0.0},
    {"col_1": "1", "col_2": 1, "col_3": 1.0},
    {"col_1": "2", "col_2": 2, "col_3": 2.0},
    {"col_1": "3", "col_2": 3, "col_3": 3.0},
]
DATA2 = [
    {"col_1": "4", "col_2": 4, "col_3": 4.0},
    {"col_1": "5", "col_2": 5, "col_3": 5.0},
]
DATA_DICT_OF_LISTS = {
    "col_1": ["0", "1", "2", "3"],
    "col_2": [0, 1, 2, 3],
    "col_3": [0.0, 1.0, 2.0, 3.0],
}

DATA_312 = [
    {"col_3": 0.0, "col_1": "0", "col_2": 0},
    {"col_3": 1.0, "col_1": "1", "col_2": 1},
]

DATA_STR = [
    {"col_1": "s0", "col_2": 0, "col_3": 0.0},
    {"col_1": "s1", "col_2": 1, "col_3": 1.0},
    {"col_1": "s2", "col_2": 2, "col_3": 2.0},
    {"col_1": "s3", "col_2": 3, "col_3": 3.0},
]


@pytest.fixture(scope="session")
def dataset_dict():
    return DATA_DICT_OF_LISTS


@pytest.fixture(scope="session")
def arrow_path(tmp_path_factory):
    import datasets

    dataset = datasets.Dataset.from_dict(DATA_DICT_OF_LISTS)
    path = str(tmp_path_factory.mktemp("data") / "dataset.arrow")
    dataset.map(cache_file_name=path)
    return path


@pytest.fixture(scope="session")
def sqlite_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("data") / "dataset.sqlite")
    with contextlib.closing(sqlite3.connect(path)) as con:
        cur = con.cursor()
        cur.execute("CREATE TABLE dataset(col_1 text, col_2 int, col_3 real)")
        for item in DATA:
            cur.execute(
                "INSERT INTO dataset(col_1, col_2, col_3) VALUES (?, ?, ?)",
                tuple(item.values()),
            )
        con.commit()
    return path


@pytest.fixture(scope="session")
def csv_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("data") / "dataset.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["col_1", "col_2", "col_3"])
        writer.writeheader()
        for item in DATA:
            writer.writerow(item)
    return path


@pytest.fixture(scope="session")
def csv2_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("data") / "dataset2.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["col_1", "col_2", "col_3"])
        writer.writeheader()
        for item in DATA:
            writer.writerow(item)
    return path


@pytest.fixture(scope="session")
def bz2_csv_path(csv_path, tmp_path_factory):
    import bz2

    path = tmp_path_factory.mktemp("data") / "dataset.csv.bz2"
    with open(csv_path, "rb") as f:
        data = f.read()
    # data = bytes(FILE_CONTENT, "utf-8")
    with bz2.open(path, "wb") as f:
        f.write(data)
    return path


@pytest.fixture(scope="session")
def zip_csv_path(csv_path, csv2_path, tmp_path_factory):
    path = tmp_path_factory.mktemp("zip_csv_path") / "csv-dataset.zip"
    with zipfile.ZipFile(path, "w") as f:
        f.write(csv_path, arcname=os.path.basename(csv_path))
        f.write(csv2_path, arcname=os.path.basename(csv2_path))
    return path


@pytest.fixture(scope="session")
def zip_uppercase_csv_path(csv_path, csv2_path, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "dataset.csv.zip"
    with zipfile.ZipFile(path, "w") as f:
        f.write(csv_path, arcname=os.path.basename(csv_path.replace(".csv", ".CSV")))
        f.write(csv2_path, arcname=os.path.basename(csv2_path.replace(".csv", ".CSV")))
    return path


@pytest.fixture(scope="session")
def zip_csv_with_dir_path(csv_path, csv2_path, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "dataset_with_dir.csv.zip"
    with zipfile.ZipFile(path, "w") as f:
        f.write(csv_path, arcname=os.path.join("main_dir", os.path.basename(csv_path)))
        f.write(
            csv2_path, arcname=os.path.join("main_dir", os.path.basename(csv2_path))
        )
    return path


@pytest.fixture(scope="session")
def parquet_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("data") / "dataset.parquet")
    schema = pa.schema(
        {
            "col_1": pa.string(),
            "col_2": pa.int64(),
            "col_3": pa.float64(),
        }
    )
    with open(path, "wb") as f:
        writer = pq.ParquetWriter(f, schema=schema)
        pa_table = pa.Table.from_pydict(
            {k: [DATA[i][k] for i in range(len(DATA))] for k in DATA[0]}, schema=schema
        )
        writer.write_table(pa_table)
        writer.close()
    return path


@pytest.fixture(scope="session")
def json_list_of_dicts_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("data") / "dataset.json")
    data = {"data": DATA}
    with open(path, "w") as f:
        json.dump(data, f)
    return path


@pytest.fixture(scope="session")
def json_dict_of_lists_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("data") / "dataset.json")
    data = {"data": DATA_DICT_OF_LISTS}
    with open(path, "w") as f:
        json.dump(data, f)
    return path


@pytest.fixture(scope="session")
def jsonl_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("data") / "dataset.jsonl")
    with open(path, "w") as f:
        for item in DATA:
            f.write(json.dumps(item) + "\n")
    return path


@pytest.fixture(scope="session")
def jsonl2_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("data") / "dataset2.jsonl")
    with open(path, "w") as f:
        for item in DATA:
            f.write(json.dumps(item) + "\n")
    return path


@pytest.fixture(scope="session")
def jsonl_312_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("data") / "dataset_312.jsonl")
    with open(path, "w") as f:
        for item in DATA_312:
            f.write(json.dumps(item) + "\n")
    return path


@pytest.fixture(scope="session")
def jsonl_str_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("data") / "dataset-str.jsonl")
    with open(path, "w") as f:
        for item in DATA_STR:
            f.write(json.dumps(item) + "\n")
    return path


@pytest.fixture(scope="session")
def text_gz_path(tmp_path_factory, text_path):
    import gzip

    path = str(tmp_path_factory.mktemp("data") / "dataset.txt.gz")
    with open(text_path, "rb") as orig_file:
        with gzip.open(path, "wb") as zipped_file:
            zipped_file.writelines(orig_file)
    return path


@pytest.fixture(scope="session")
def jsonl_gz_path(tmp_path_factory, jsonl_path):
    import gzip

    path = str(tmp_path_factory.mktemp("data") / "dataset.jsonl.gz")
    with open(jsonl_path, "rb") as orig_file:
        with gzip.open(path, "wb") as zipped_file:
            zipped_file.writelines(orig_file)
    return path


@pytest.fixture(scope="session")
def zip_jsonl_path(jsonl_path, jsonl2_path, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "dataset.jsonl.zip"
    with zipfile.ZipFile(path, "w") as f:
        f.write(jsonl_path, arcname=os.path.basename(jsonl_path))
        f.write(jsonl2_path, arcname=os.path.basename(jsonl2_path))
    return path


@pytest.fixture(scope="session")
def zip_nested_jsonl_path(zip_jsonl_path, jsonl_path, jsonl2_path, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "dataset_nested.jsonl.zip"
    with zipfile.ZipFile(path, "w") as f:
        f.write(
            zip_jsonl_path,
            arcname=os.path.join("nested", os.path.basename(zip_jsonl_path)),
        )
    return path


@pytest.fixture(scope="session")
def zip_jsonl_with_dir_path(jsonl_path, jsonl2_path, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "dataset_with_dir.jsonl.zip"
    with zipfile.ZipFile(path, "w") as f:
        f.write(
            jsonl_path, arcname=os.path.join("main_dir", os.path.basename(jsonl_path))
        )
        f.write(
            jsonl2_path, arcname=os.path.join("main_dir", os.path.basename(jsonl2_path))
        )
    return path


@pytest.fixture(scope="session")
def tar_jsonl_path(jsonl_path, jsonl2_path, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "dataset.jsonl.tar"
    with tarfile.TarFile(path, "w") as f:
        f.add(jsonl_path, arcname=os.path.basename(jsonl_path))
        f.add(jsonl2_path, arcname=os.path.basename(jsonl2_path))
    return path


@pytest.fixture(scope="session")
def tar_nested_jsonl_path(tar_jsonl_path, jsonl_path, jsonl2_path, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "dataset_nested.jsonl.tar"
    with tarfile.TarFile(path, "w") as f:
        f.add(
            tar_jsonl_path,
            arcname=os.path.join("nested", os.path.basename(tar_jsonl_path)),
        )
    return path


@pytest.fixture(scope="session")
def text_path(tmp_path_factory):
    data = ["0", "1", "2", "3"]
    path = str(tmp_path_factory.mktemp("data") / "dataset.txt")
    with open(path, "w") as f:
        for item in data:
            f.write(item + "\n")
    return path


@pytest.fixture(scope="session")
def text2_path(tmp_path_factory):
    data = ["0", "1", "2", "3"]
    path = str(tmp_path_factory.mktemp("data") / "dataset2.txt")
    with open(path, "w") as f:
        for item in data:
            f.write(item + "\n")
    return path


@pytest.fixture(scope="session")
def text_dir(tmp_path_factory):
    data = ["0", "1", "2", "3"]
    path = tmp_path_factory.mktemp("data_text_dir") / "dataset.txt"
    with open(path, "w") as f:
        for item in data:
            f.write(item + "\n")
    return path.parent


@pytest.fixture(scope="session")
def text_dir_with_unsupported_extension(tmp_path_factory):
    data = ["0", "1", "2", "3"]
    path = tmp_path_factory.mktemp("data") / "dataset.abc"
    with open(path, "w") as f:
        for item in data:
            f.write(item + "\n")
    return path


@pytest.fixture(scope="session")
def zip_text_path(text_path, text2_path, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "dataset.text.zip"
    with zipfile.ZipFile(path, "w") as f:
        f.write(text_path, arcname=os.path.basename(text_path))
        f.write(text2_path, arcname=os.path.basename(text2_path))
    return path


@pytest.fixture(scope="session")
def zip_text_with_dir_path(text_path, text2_path, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "dataset_with_dir.text.zip"
    with zipfile.ZipFile(path, "w") as f:
        f.write(
            text_path, arcname=os.path.join("main_dir", os.path.basename(text_path))
        )
        f.write(
            text2_path, arcname=os.path.join("main_dir", os.path.basename(text2_path))
        )
    return path


@pytest.fixture(scope="session")
def zip_unsupported_ext_path(text_path, text2_path, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "dataset.ext.zip"
    with zipfile.ZipFile(path, "w") as f:
        f.write(text_path, arcname=os.path.basename("unsupported.ext"))
        f.write(text2_path, arcname=os.path.basename("unsupported_2.ext"))
    return path


@pytest.fixture(scope="session")
def text_path_with_unicode_new_lines(tmp_path_factory):
    text = "\n".join(["First", "Second\u2029with Unicode new line", "Third"])
    path = str(tmp_path_factory.mktemp("data") / "dataset_with_unicode_new_lines.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


@pytest.fixture(scope="session")
def image_file():
    return os.path.join("tests", "features", "data", "test_image_rgb.jpg")


@pytest.fixture(scope="session")
def audio_file():
    return os.path.join("tests", "features", "data", "test_audio_44100.wav")


@pytest.fixture(scope="session")
def bio_dir():
    return os.path.join("tests", "features", "data", "CAMDA")


@pytest.fixture(scope="session")
def metadata_bio_files():
    return os.path.join("tests", "features", "data", "CAMDA", "camda.pheno.csv")


@pytest.fixture(scope="session")
def anno_bio_files():
    return os.path.join("tests", "features", "data", "CAMDA", "camda.feature.csv")


@pytest.fixture(scope="session")
def zip_image_path(image_file, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "dataset.img.zip"
    with zipfile.ZipFile(path, "w") as f:
        f.write(image_file, arcname=os.path.basename(image_file))
        f.write(
            image_file, arcname=os.path.basename(image_file).replace(".jpg", "2.jpg")
        )
    return path


@pytest.fixture(scope="session")
def data_dir_with_hidden_files(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data_dir")

    (data_dir / "subdir").mkdir()
    with open(data_dir / "subdir" / "train.txt", "w") as f:
        f.write("foo\n" * 10)
    with open(data_dir / "subdir" / "test.txt", "w") as f:
        f.write("bar\n" * 10)
    # hidden file
    with open(data_dir / "subdir" / ".test.txt", "w") as f:
        f.write("bar\n" * 10)

    # hidden directory
    (data_dir / ".subdir").mkdir()
    with open(data_dir / ".subdir" / "train.txt", "w") as f:
        f.write("foo\n" * 10)
    with open(data_dir / ".subdir" / "test.txt", "w") as f:
        f.write("bar\n" * 10)

    return data_dir
