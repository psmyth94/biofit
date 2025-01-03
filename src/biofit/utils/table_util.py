import copy
import inspect
import os
import re
import tempfile
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import List, Union

import pyarrow as pa
from biocore.utils.import_util import (
    is_datasets_available,
    is_rpy2_arrow_available,
    is_rpy2_available,
    requires_backends,
)

from biofit import config

from . import logging
from .file_utils import move_temp_file

logger = logging.get_logger(__name__)


# Function adapted from the Hugging Face Team, Copyright 2024 The Hugging Face Team
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Modified by: Patrick Smyth
# Date: 2024
# Summary of changes:
# - Changed names from pyarrow to pa for consistency
def _arrow_to_datasets_dtype(arrow_type: pa.DataType) -> str:
    """
    _arrow_to_datasets_dtype takes a pa.DataType and converts it to a datasets string dtype.
    In effect, `dt == string_to_arrow(_arrow_to_datasets_dtype(dt))`
    """
    if pa.types.is_null(arrow_type):
        return "null"
    elif pa.types.is_boolean(arrow_type):
        return "bool"
    elif pa.types.is_int8(arrow_type):
        return "int8"
    elif pa.types.is_int16(arrow_type):
        return "int16"
    elif pa.types.is_int32(arrow_type):
        return "int32"
    elif pa.types.is_int64(arrow_type):
        return "int64"
    elif pa.types.is_uint8(arrow_type):
        return "uint8"
    elif pa.types.is_uint16(arrow_type):
        return "uint16"
    elif pa.types.is_uint32(arrow_type):
        return "uint32"
    elif pa.types.is_uint64(arrow_type):
        return "uint64"
    elif pa.types.is_float16(arrow_type):
        return "float16"  # pa dtype is "halffloat"
    elif pa.types.is_float32(arrow_type):
        return "float32"  # pa dtype is "float"
    elif pa.types.is_float64(arrow_type):
        return "float64"  # pa dtype is "double"
    elif pa.types.is_time32(arrow_type):
        return f"time32[{pa.type_for_alias(str(arrow_type)).unit}]"
    elif pa.types.is_time64(arrow_type):
        return f"time64[{pa.type_for_alias(str(arrow_type)).unit}]"
    elif pa.types.is_timestamp(arrow_type):
        if arrow_type.tz is None:
            return f"timestamp[{arrow_type.unit}]"
        elif arrow_type.tz:
            return f"timestamp[{arrow_type.unit}, tz={arrow_type.tz}]"
        else:
            raise ValueError(f"Unexpected timestamp object {arrow_type}.")
    elif pa.types.is_date32(arrow_type):
        return "date32"  # pa dtype is "date32[day]"
    elif pa.types.is_date64(arrow_type):
        return "date64"  # pa dtype is "date64[ms]"
    elif pa.types.is_duration(arrow_type):
        return f"duration[{arrow_type.unit}]"
    elif pa.types.is_decimal128(arrow_type):
        return f"decimal128({arrow_type.precision}, {arrow_type.scale})"
    elif pa.types.is_decimal256(arrow_type):
        return f"decimal256({arrow_type.precision}, {arrow_type.scale})"
    elif pa.types.is_binary(arrow_type):
        return "binary"
    elif pa.types.is_large_binary(arrow_type):
        return "large_binary"
    elif pa.types.is_string(arrow_type):
        return "string"
    elif pa.types.is_large_string(arrow_type):
        return "large_string"
    elif pa.types.is_dictionary(arrow_type):
        return _arrow_to_datasets_dtype(arrow_type.value_type)
    else:
        raise ValueError(
            f"Arrow type {arrow_type} does not have a datasets dtype equivalent."
        )


# Function adapted from the Hugging Face Team, Copyright 2024 The Hugging Face Team
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
def string_to_arrow(datasets_dtype: str) -> pa.DataType:
    """
    string_to_arrow takes a datasets string dtype and converts it to a pyarrow.DataType.

    In effect, `dt == string_to_arrow(_arrow_to_datasets_dtype(dt))`

    This is necessary because the datasets.Value() primitive type is constructed using a string dtype

    Value(dtype=str)

    But Features.type (via `get_nested_type()` expects to resolve Features into a pyarrow Schema,
        which means that each Value() must be able to resolve into a corresponding pyarrow.DataType, which is the
        purpose of this function.
    """

    def _dtype_error_msg(dtype, pa_dtype, examples=None, urls=None):
        msg = f"{dtype} is not a validly formatted string representation of the pyarrow {pa_dtype} type."
        if examples:
            examples = (
                ", ".join(examples[:-1]) + " or " + examples[-1]
                if len(examples) > 1
                else examples[0]
            )
            msg += f"\nValid examples include: {examples}."
        if urls:
            urls = (
                ", ".join(urls[:-1]) + " and " + urls[-1] if len(urls) > 1 else urls[0]
            )
            msg += f"\nFor more insformation, see: {urls}."
        return msg

    if datasets_dtype in pa.__dict__:
        return pa.__dict__[datasets_dtype]()

    if (datasets_dtype + "_") in pa.__dict__:
        return pa.__dict__[datasets_dtype + "_"]()

    timestamp_matches = re.search(r"^timestamp\[(.*)\]$", datasets_dtype)
    if timestamp_matches:
        timestamp_internals = timestamp_matches.group(1)
        internals_matches = re.search(
            r"^(s|ms|us|ns),\s*tz=([a-zA-Z0-9/_+\-:]*)$", timestamp_internals
        )
        if timestamp_internals in ["s", "ms", "us", "ns"]:
            return pa.timestamp(timestamp_internals)
        elif internals_matches:
            return pa.timestamp(internals_matches.group(1), internals_matches.group(2))
        else:
            raise ValueError(
                _dtype_error_msg(
                    datasets_dtype,
                    "timestamp",
                    examples=["timestamp[us]", "timestamp[us, tz=America/New_York"],
                    urls=[
                        "https://arrow.apache.org/docs/python/generated/pyarrow.timestamp.html"
                    ],
                )
            )

    duration_matches = re.search(r"^duration\[(.*)\]$", datasets_dtype)
    if duration_matches:
        duration_internals = duration_matches.group(1)
        if duration_internals in ["s", "ms", "us", "ns"]:
            return pa.duration(duration_internals)
        else:
            raise ValueError(
                _dtype_error_msg(
                    datasets_dtype,
                    "duration",
                    examples=["duration[s]", "duration[us]"],
                    urls=[
                        "https://arrow.apache.org/docs/python/generated/pyarrow.duration.html"
                    ],
                )
            )

    time_matches = re.search(r"^time(.*)\[(.*)\]$", datasets_dtype)
    if time_matches:
        time_internals_bits = time_matches.group(1)
        if time_internals_bits == "32":
            time_internals_unit = time_matches.group(2)
            if time_internals_unit in ["s", "ms"]:
                return pa.time32(time_internals_unit)
            else:
                raise ValueError(
                    f"{time_internals_unit} is not a valid unit for the pyarrow time32 type. Supported units: s (second) and ms (millisecond)."
                )
        elif time_internals_bits == "64":
            time_internals_unit = time_matches.group(2)
            if time_internals_unit in ["us", "ns"]:
                return pa.time64(time_internals_unit)
            else:
                raise ValueError(
                    f"{time_internals_unit} is not a valid unit for the pyarrow time64 type. Supported units: us (microsecond) and ns (nanosecond)."
                )
        else:
            raise ValueError(
                _dtype_error_msg(
                    datasets_dtype,
                    "time",
                    examples=["time32[s]", "time64[us]"],
                    urls=[
                        "https://arrow.apache.org/docs/python/generated/pyarrow.time32.html",
                        "https://arrow.apache.org/docs/python/generated/pyarrow.time64.html",
                    ],
                )
            )

    decimal_matches = re.search(r"^decimal(.*)\((.*)\)$", datasets_dtype)
    if decimal_matches:
        decimal_internals_bits = decimal_matches.group(1)
        if decimal_internals_bits == "128":
            decimal_internals_precision_and_scale = re.search(
                r"^(\d+),\s*(-?\d+)$", decimal_matches.group(2)
            )
            if decimal_internals_precision_and_scale:
                precision = decimal_internals_precision_and_scale.group(1)
                scale = decimal_internals_precision_and_scale.group(2)
                return pa.decimal128(int(precision), int(scale))
            else:
                raise ValueError(
                    _dtype_error_msg(
                        datasets_dtype,
                        "decimal128",
                        examples=["decimal128(10, 2)", "decimal128(4, -2)"],
                        urls=[
                            "https://arrow.apache.org/docs/python/generated/pyarrow.decimal128.html"
                        ],
                    )
                )
        elif decimal_internals_bits == "256":
            decimal_internals_precision_and_scale = re.search(
                r"^(\d+),\s*(-?\d+)$", decimal_matches.group(2)
            )
            if decimal_internals_precision_and_scale:
                precision = decimal_internals_precision_and_scale.group(1)
                scale = decimal_internals_precision_and_scale.group(2)
                return pa.decimal256(int(precision), int(scale))
            else:
                raise ValueError(
                    _dtype_error_msg(
                        datasets_dtype,
                        "decimal256",
                        examples=["decimal256(30, 2)", "decimal256(38, -4)"],
                        urls=[
                            "https://arrow.apache.org/docs/python/generated/pyarrow.decimal256.html"
                        ],
                    )
                )
        else:
            raise ValueError(
                _dtype_error_msg(
                    datasets_dtype,
                    "decimal",
                    examples=["decimal128(12, 3)", "decimal256(40, 6)"],
                    urls=[
                        "https://arrow.apache.org/docs/python/generated/pyarrow.decimal128.html",
                        "https://arrow.apache.org/docs/python/generated/pyarrow.decimal256.html",
                    ],
                )
            )

    raise ValueError(
        f"Neither {datasets_dtype} nor {datasets_dtype + '_'} seems to be a pyarrow data type. "
        f"Please make sure to use a correct data type, see: "
        f"https://arrow.apache.org/docs/python/api/datatypes.html#factory-functions"
    )


def is_binary_like(data_type: pa.DataType) -> bool:
    return pa.types.is_binary(data_type) or pa.types.is_unicode(data_type)


def is_large_binary_like(data_type: pa.DataType) -> bool:
    return pa.types.is_large_binary(data_type) or pa.types.is_large_unicode(data_type)


def is_fixed_width(data_type: pa.DataType) -> bool:
    return (
        pa.types.is_primitive(data_type)
        or pa.types.is_dictionary(data_type)
        or pa.types.is_large_list(data_type)
    )


def upcast_tables(tables: List[pa.Table]):
    cols = None
    cols_diff_dtypes = defaultdict(set)
    for table in tables:
        if cols is None:
            cols = {col.name: col.type for col in table.schema}
        else:
            cols.update(
                {col.name: col.type for col in table.schema if col.name not in cols}
            )
            for col in table.schema:
                if cols[col.name] != col.type:
                    cols_diff_dtypes[col.name].update({col.type, cols[col.name]})
    cols_to_cast = {}
    for col, types in cols_diff_dtypes.items():
        types = [_arrow_to_datasets_dtype(t) for t in types]
        cols_to_cast[col] = string_to_arrow(determine_upcast(types))

    new_tables = []
    for table in tables:
        cols = table.schema.names
        casts = pa.schema(
            [
                (col.name, cols_to_cast[col.name]) if col.name in cols_to_cast else col
                for col in table.schema
            ]
        )
        new_tables.append(table.cast(casts))
    return new_tables


def determine_upcast(dtype_list):
    """
    Determines the upcasted data type from a list of data types based on a predefined hierarchy.

    Args:
        dtype_list (list): List of data types to be considered for upcasting.

    Returns:
        str: The upcasted data type.
    """

    # Define the hierarchy of dtypes with their priority levels
    dtype_hierarchy = {
        "null": 0,
        "bool": 1,
        "int8": 2,
        "int16": 3,
        "int32": 4,
        "int": 5,
        "int64": 5,
        "uint8": 6,
        "uint16": 7,
        "uint32": 8,
        "uint64": 9,
        "float16": 10,
        "float": 11,
        "float32": 11,
        "float64": 12,
        "time32[s]": 13,
        "time32[ms]": 14,
        "time64[us]": 15,
        "time64[ns]": 16,
        "timestamp[s]": 17,
        "timestamp[ms]": 18,
        "timestamp[us]": 19,
        "timestamp[ns]": 20,
        "date32": 21,
        "date64": 22,
        "duration[s]": 23,
        "duration[ms]": 24,
        "duration[us]": 25,
        "duration[ns]": 26,
        "decimal128": 27,
        "decimal256": 28,
        "binary": 29,
        "large_binary": 30,
        "string": 31,
        "large_string": 32,
    }

    highest_priority = 0
    upcast_dtype = "null"  # the lowest in hierarchy

    for dtype in dtype_list:
        priority = dtype_hierarchy.get(dtype, None)
        if not priority:
            raise ValueError(f"Invalid dtype found {dtype}")
        if priority > highest_priority:
            highest_priority = priority
            upcast_dtype = dtype

    return upcast_dtype


def concat_blocks(pa_tables: List[pa.Table], axis: int = 0, append=True) -> pa.Table:
    if axis == 0:
        # we set promote=True to fill missing columns with null values
        if config.PYARROW_VERSION.major < 14:
            return pa.concat_tables(pa_tables, promote=True)
        else:
            return pa.concat_tables(pa_tables, promote_options="permissive")
    elif axis == 1:
        for i, table in enumerate(pa_tables):
            if i == 0:
                pa_table = table
            else:
                for name, col in zip(table.column_names, table.columns):
                    if append:
                        pa_table = pa_table.append_column(name, col)
                    else:
                        pa_table = pa_table.add_column(0, name, col)
        return pa_table
    else:
        raise ValueError("'axis' must be either 0 or 1")


def _arrow_join(
    left_table: pa.Table,
    right_table: pa.Table,
    keys: Union[str, List[str]],
    right_keys: Union[str, List[str]] = None,
    join_type="left outer",
    left_suffix=None,
    right_suffix=None,
    coalesce_keys=True,
    use_threads=True,
) -> pa.Table:
    """Extends arrow's join to support joining on struct columns at any nested level.

    Args:
        right_table (`Table`):
            The table to join to the current one, acting as the right table in the join operation.

        keys (`Union[str, List[str]]`):
            The columns from current table that should be used as keys of the join operation left side.

        right_keys (`Union[str, List[str]]`, *optional*):
            The columns from the right_table that should be used as keys on the join operation right side. When None use the same key names as the left table.

        join_type (`str`, Defaults to 'left outer'):
            The kind of join that should be performed, one of (“left semi”, “right semi”, “left anti”, “right anti”, “inner”, “left outer”, “right outer”, “full outer”)

        left_suffix (`str`, *optional*):
            Which suffix to add to left column names. This prevents confusion when the columns in left and right tables have colliding names.

        right_suffix (`str`, *optional*):
            Which suffix to add to the right column names. This prevents confusion when the columns in left and right tables have colliding names.

        coalesce_keys (`bool`, Defaults to True):
            If the duplicated keys should be omitted from one of the sides in the join result.

        use_threads (`bool`, Defaults to True):
            Whether to use multithreading or not.


    """

    if isinstance(keys, str):
        keys = [keys]

    if right_keys is None:
        right_keys = keys
    else:
        if isinstance(right_keys, str):
            right_keys = [right_keys]

    left_cast = []
    right_cast = []

    def _get_struct_columns_and_prepare_cast(
        left_schema: pa.Schema, right_schema: pa.Schema
    ) -> dict:
        struct_columns = []
        keys = set()

        def process_nested_dict(key, nested_schema):
            if key in keys:
                return
            keys.add(key)
            # Check if the value is a list of dictionaries
            if pa.types.is_struct(nested_schema):
                full_keys = []
                original_keys = []
                for nested_field in nested_schema:
                    # Recursively process the nested dictionary
                    original_keys.append(nested_field.name)
                    full_key = f"{key}.{nested_field.name}"
                    full_keys.append(full_key)
                    process_nested_dict(full_key, nested_field.type)
                struct_columns.append((key, original_keys, full_keys))

        for field in left_schema:
            if pa.types.is_struct(field.type):
                left_cast.append(pa.field(field.name, field.type))
                process_nested_dict(field.name, field.type)
            elif (
                pa.types.is_list(field.type)
                or pa.types.is_large_list(field.type)
                or pa.types.is_fixed_size_list(field.type)
            ):
                raise pa.ArrowNotImplementedError(
                    "Joining on lists is not supported. Please load them as a struct before joining. For example, change the column with value [1,2,3] to {'a': 1, 'b': 2, 'c': 3}"
                )
            elif pa.types.is_null(field.type):
                left_cast.append(pa.field(field.name, pa.string()))
            elif (
                not pa.types.is_fixed_size_list
                and not is_fixed_width(field.type)
                and not is_binary_like(field.type)
                and not is_large_binary_like(field.type)
            ):
                left_cast.append(pa.field(field.name, pa.string()))
            else:
                left_cast.append(pa.field(field.name, field.type))

        for field in right_schema:
            if pa.types.is_struct(field.type):
                right_cast.append(pa.field(field.name, field.type))
                process_nested_dict(field.name, field.type)
            elif (
                pa.types.is_list(field.type)
                or pa.types.is_large_list(field.type)
                or pa.types.is_fixed_size_list(field.type)
            ):
                raise pa.ArrowNotImplementedError(
                    "Joining on lists is not supported. Please load them as a struct before joining. For example, change the column with value [1,2,3] to {'a': 1, 'b': 2, 'c': 3}"
                )
            elif pa.types.is_null(field.type):
                right_cast.append(pa.field(field.name, pa.string()))
            elif (
                not pa.types.is_fixed_size_list
                and not is_fixed_width(field.type)
                and not is_binary_like(field.type)
                and not is_large_binary_like(field.type)
            ):
                right_cast.append(pa.field(field.name, pa.string()))
            else:
                right_cast.append(pa.field(field.name, field.type))

        return struct_columns

    def reconstruct_table(joined_table, struct_cols):
        """
        Reconstruct struct columns from flattened columns based on original schema.
        """
        for column, orig_names, nested_columns in struct_cols:
            nested_data = {
                sub_col: joined_table[col].combine_chunks()
                for col, sub_col in zip(nested_columns, orig_names)
            }
            reconstructed_nested_col = pa.StructArray.from_arrays(
                nested_data.values(), nested_data.keys()
            )
            index = joined_table.schema.get_field_index(nested_columns[0])
            joined_table = joined_table.drop(nested_columns).add_column(
                index, column, reconstructed_nested_col
            )
        return joined_table

    def get_nested_level(schema, level=0):
        """
        Get the maximum level of nesting in a struct schema.
        """
        max_level = level
        for field in schema:
            if pa.types.is_struct(field.type):
                max_level = max(
                    max_level, get_nested_level(field.type, level=level + 1)
                )
        return max_level

    def flatten_table(table, max_level, current_level=0):
        """
        Recursively flatten a table.
        """
        if current_level == max_level:
            return table
        return flatten_table(
            table.flatten(), max_level, current_level=current_level + 1
        )

    left_nested_level = get_nested_level(left_table.schema)
    right_nested_level = get_nested_level(right_table.schema)
    struct_columns = _get_struct_columns_and_prepare_cast(
        left_table.schema, right_table.schema
    )

    colliding_names = list(
        (set(right_table.column_names) & set(left_table.column_names))
        - set(keys)
        - set(right_keys)
    )

    for left_key, right_key in zip(keys, right_keys):
        if left_table[left_key].type != right_table[right_key].type:
            index = right_table.schema.get_field_index(right_key)
            right_cast[index] = pa.field(right_key, left_table[left_key].type)

    left_table = left_table.cast(pa.schema(left_cast)) if left_cast else left_table
    right_table = right_table.cast(pa.schema(right_cast)) if right_cast else right_table

    return reconstruct_table(
        flatten_table(left_table, left_nested_level).join(
            flatten_table(right_table.drop(colliding_names), right_nested_level),
            keys=keys,
            right_keys=right_keys,
            join_type=join_type,
            left_suffix=left_suffix,
            right_suffix=right_suffix,
            coalesce_keys=coalesce_keys,
            use_threads=use_threads,
        ),
        struct_columns,
    )


def init_arrow_buffer_and_writer(
    cache_file_name,
    fingerprint=None,
    features=None,
    writer_batch_size=None,
    keep_in_memory=False,
    disable_nullable=False,
):
    if isinstance(cache_file_name, Path):
        cache_file_name = cache_file_name.resolve().as_posix()
    if is_datasets_available():
        from datasets import arrow_writer

        # Prepare output buffer and batched writer in memory or on file if we update the table
        if keep_in_memory or cache_file_name is None:
            buf_writer = pa.BufferOutputStream()
            tmp_file = None
            writer = arrow_writer.ArrowWriter(
                features=features,
                stream=buf_writer,
                writer_batch_size=writer_batch_size,
                update_features=False,
                fingerprint=fingerprint,
                disable_nullable=disable_nullable,
            )
        else:
            buf_writer = None
            tmp_file = tempfile.NamedTemporaryFile(
                "wb", dir=os.path.dirname(cache_file_name), delete=False
            )
            writer = arrow_writer.ArrowWriter(
                features=features,
                path=tmp_file.name,
                writer_batch_size=writer_batch_size,
                update_features=False,
                fingerprint=fingerprint,
                disable_nullable=disable_nullable,
            )
    else:
        buf_writer = None
        tmp_file = tempfile.NamedTemporaryFile(
            "wb", dir=os.path.dirname(cache_file_name), delete=False
        )
        if features is not None and isinstance(features, dict):
            features = pa.schema([(k, string_to_arrow(v)) for k, v in features.items()])
        if features is not None and not isinstance(features, pa.Schema):
            raise ValueError(
                "features must be a dict or a pa.Schema when datasets is not installed"
            )
        writer = pa.ipc.RecordBatchStreamWriter(tmp_file, schema=features)

        def finalize():
            writer.close()

        writer.finalize = finalize
    return buf_writer, writer, tmp_file


def write_arrow_table(
    table: pa.Table,
    cache_file_name: Union[str, Path],
    fingerprint=None,
    features=None,
    writer_batch_size=None,
    disable_nullable=False,
):
    # requires_backends("write_arrow_table", "datasets")
    from datasets import Features

    if features is None:
        if isinstance(table, (pa.ChunkedArray, pa.Array)):
            data_type = table.type
            features = _arrow_to_datasets_dtype(data_type)
        features = Features.from_arrow_schema(table.schema)
    tmp_file = None

    try:
        _, writer, tmp_file = init_arrow_buffer_and_writer(
            cache_file_name,
            fingerprint=fingerprint,
            features=features,
            writer_batch_size=writer_batch_size,
            disable_nullable=disable_nullable,
        )
        writer.write_table(table)
        writer.finalize()
    except:
        if tmp_file and os.path.exists(tmp_file.name):
            tmp_file.close()
            os.remove(tmp_file.name)
        raise
    return move_temp_file(tmp_file, cache_file_name)


def python_to_r(value):
    """Converts a Python primitive type into its R equivalent."""

    if value is None:
        return "NULL"

    elif isinstance(value, bool):
        return "TRUE" if value else "FALSE"

    elif isinstance(value, (int, float)):
        return str(value)

    elif isinstance(value, complex):
        return f"complex(real = {value.real}, imaginary = {value.imag})"

    elif isinstance(value, str):
        return f'"{value}"'  # Add quotes around strings for R

    elif isinstance(value, list) or isinstance(value, tuple):
        # Recursively convert elements of the list/tuple to R format
        converted_elements = ", ".join(python_to_r(elem) for elem in value)
        return f"c({converted_elements})"

    elif isinstance(value, dict):
        # Convert dict to named list in R
        converted_items = [
            f"{python_to_r(k)} = {python_to_r(v)}" for k, v in value.items()
        ]
        return f"list({', '.join(converted_items)})"

    elif isinstance(value, set):
        # Convert set to unique vector in R
        converted_elements = ", ".join(python_to_r(elem) for elem in sorted(value))
        return f"unique(c({converted_elements}))"

    elif isinstance(value, range):
        # Convert range to sequence in R
        return f"seq({value.start}, {value.stop - 1}, by = {value.step})"

    elif isinstance(value, bytes):
        # Handle bytes conversion to raw in R (if needed, otherwise leave as unsupported)
        return f"as.raw(c({', '.join(hex(b) for b in value)}))"

    else:
        raise TypeError(f"Type {type(value)} is not supported.")


def debug_r_script(path):
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    path = path.as_posix()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **func_kwargs):
            name = func.__name__
            if is_rpy2_arrow_available():
                from rpy2.robjects import ListVector, default_converter
                from rpy2_arrow.arrow import converter

                _converter = default_converter + converter
            elif is_rpy2_available():
                from rpy2.robjects import ListVector, conversion

                _converter = (
                    conversion.get_conversion()
                    if getattr(conversion, "get_conversion", None)
                    else conversion.converter
                )
            else:
                # suggest installing rpy2_arrow if rpy2 is not available
                requires_backends(name, "rpy2_arrow")

            import rpy2.rinterface
            import rpy2.robjects as ro
            from rpy2.rinterface import Sexp

            self = args[0]
            if hasattr(self, "plotter"):
                method_args = [
                    p.name
                    for p in inspect.signature(self.plotter).parameters.values()
                    if p != p.VAR_KEYWORD
                ]
            else:
                method_args = [
                    p.name
                    for p in inspect.signature(func).parameters.values()
                    if p != p.VAR_KEYWORD
                ][1:]

            kwargs = copy.deepcopy(func_kwargs)
            for i, arg in enumerate(args[1:]):
                kwargs[method_args[i]] = arg

            kwargs.update(self.config.get_params())

            def convert_to_r(arg):
                if isinstance(arg, Sexp):
                    return arg
                elif arg is None:
                    return ro.NULL
                elif isinstance(arg, (list, tuple)):
                    return _converter.py2rpy([convert_to_r(a) for a in arg])
                elif isinstance(arg, dict):
                    return ListVector(arg)
                else:
                    return _converter.py2rpy(arg)

            debug_script = "options(error = traceback)\n"
            debug_script += (
                f'R_SCRIPTS_PATH <- "{config.R_SCRIPTS.resolve().as_posix()}"\n'
            )
            debug_script += 'source(file.path(R_SCRIPTS_PATH, "plotting_utils.R"))\n'
            debug_script += 'require("arrow")\n'
            tmp_dir = tempfile.gettempdir()
            with rpy2.rinterface.local_context() as r_context:
                save_vars = []
                for k, v in kwargs.items():
                    if isinstance(v, (pa.Array, pa.Table, pa.ChunkedArray)):
                        cache_file_name = f"{tmp_dir}/{k}.arrow"
                        write_arrow_table(v, cache_file_name)
                        debug_script += (
                            f"{k} <- as.data.frame(arrow::read_ipc_stream("
                            f"'{cache_file_name}', as_data_frame = FALSE))\n"
                        )
                    else:
                        try:
                            debug_script += f"{k} <- {python_to_r(v)}\n"
                        except TypeError:
                            save_vars.append(k)
                            r_context[k] = convert_to_r(arg)
                if save_vars:
                    code = ", ".join(r_context.keys())
                    code = f"save({code}, file = '{tmp_dir}/data.RData')"
                    ro.r(code)
                    debug_script = debug_script + f"load('{tmp_dir}/data.RData')\n"
            script = self.r_caller.r_code.split("\n")
            first_start = False
            second_start = False
            for line in script:
                if first_start and second_start:
                    debug_script += f"{line[2:]}\n"
                if line.startswith(self.config.main_method):
                    first_start = True
                if first_start and line.endswith("{"):
                    second_start = True
                elif line.startswith("}") and first_start and second_start:
                    break
            with open(f"{path}/debug.R", "w") as f:
                f.write(debug_script)
            return func(*args, **func_kwargs)

        return wrapper

    return decorator


def read_arrow_table(
    cache_file_name: Union[str, Path],
):
    return pa.ipc.open_stream(cache_file_name).read_all()
