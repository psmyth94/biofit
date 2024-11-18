from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Type, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from biocore import DataHandler

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes
from biofit.utils import logging

from ..filtering import SampleFilter, SampleFilterConfig

if TYPE_CHECKING:
    import polars as pl

logger = logging.get_logger(__name__)


@dataclass
class MissingLabelsSampleFilterConfig(SampleFilterConfig):
    # process description
    _transform_process_desc: str = field(
        default="SampleFilter out rows with missing labels", init=False, repr=False
    )
    processor_name: str = field(default="missing_labels", init=False, repr=False)
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("TARGET_FEATURE_TYPES")],
        init=False,
        repr=False,
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("TARGET_FEATURE_TYPES")],
        init=False,
        repr=False,
    )

    missing_label: Optional[Union[str, int]] = "auto"

    def __post_init__(self):
        self._transform_process_desc = (
            f"SampleFiltering out rows with labels equaling to {self.missing_label}"
        )


class MissingLabelsSampleFilter(SampleFilter):
    """Remove samples that are not labeled

    - config:
    - mising_label: The value we deem as missing and want to remove. Default is "auto".
    """

    # main config class
    config_class = MissingLabelsSampleFilterConfig
    config: MissingLabelsSampleFilterConfig

    def __init__(
        self,
        config: Optional[MissingLabelsSampleFilterConfig] = None,
        missing_label: Optional[Union[str, int]] = "auto",
        **kwargs,
    ):
        super().__init__(config=config, missing_label=missing_label, **kwargs)

    def fit(
        self,
        X,
        input_columns: SelectedColumnTypes = None,
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
    ) -> "MissingLabelsSampleFilter":
        self.config._input_columns = self._set_input_columns_and_arity(input_columns)
        return self._process_fit(
            X,
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
        *args,
        input_columns: SelectedColumnTypes = None,
        keep_unused_columns: bool = True,
        raise_if_missing: bool = False,
        cache_output: bool = None,
        load_from_cache_file: bool = None,
        batched: bool = True,
        batch_size: int = 1000,
        batch_format: str = None,
        output_format: str = None,
        map_kwargs: dict = {"fn_kwargs": {}},
        num_proc: int = None,
        cache_dir: str = None,
        cache_file_name: str = None,
        fingerprint: str = None,
    ):
        return self.fit(
            X,
            input_columns=input_columns,
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

    def _process_transform_input(self, X, **kwargs):
        selected_indices = kwargs["fn_kwargs"].get("selected_indices", None) or [0]
        if self.config.missing_label == "auto":
            if DataHandler.is_categorical(X, selected_indices[0], threshold=30):
                self.config.missing_label = -1
                kwargs["desc"] = self._transform_process_desc = (
                    "SampleFiltering out rows with labels equaling to -1"
                )
            else:
                self.config.missing_label = None
                kwargs["desc"] = self._transform_process_desc = (
                    "SampleFiltering out rows with labels equaling to None"
                )
        return super()._process_transform_input(X, **kwargs)

    def _transform_arrow(self, X: Union[pa.Table, pa.Array]):
        if isinstance(X, pa.Table):
            return (
                pc.not_equal(X.column(0), self.config.missing_label).to_numpy().tolist()
            )
        return pc.not_equal(X, self.config.missing_label).to_numpy().tolist()

    def _transform_pandas(self, X: pd.DataFrame):
        return X.ne(self.config.missing_label).values.tolist()

    def _transform_polars(self, X: "pl.DataFrame"):
        import polars as pl

        if isinstance(X, pl.Series):
            return X.ne(self.config.missing_label).to_numpy().tolist()
        return (
            X.get_column(X.columns[0]).ne(self.config.missing_label).to_numpy().tolist()
        )

    def _transform_numpy(self, X: np.ndarray):
        return X != self.config.missing_label
