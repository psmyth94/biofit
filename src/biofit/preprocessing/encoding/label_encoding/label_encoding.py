from dataclasses import dataclass, field
from typing import List, Type

import numpy as np
from biocore import DataHandler

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes, sync_backup_config
from biofit.utils import logging

from ..encoding import Encoder, EncoderConfig

logger = logging.get_logger(__name__)


@dataclass
class LabelEncoderConfig(EncoderConfig):
    """Configuration class for label encoding."""

    _fit_process_desc: str = field(
        default="Determining unique labels", init=False, repr=False
    )
    _fit_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("ClassLabel")], init=False, repr=False
    )
    _transform_input_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("ClassLabel")], init=False, repr=False
    )

    processor_name: str = field(default="label_encoding", init=False, repr=False)
    _transform_process_desc: str = field(default=None, init=False, repr=False)

    names: list = field(default_factory=list, init=False, repr=False)

    # auto attributes
    label_mapping: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.names:
            self._transform_process_desc = f"Encoding labels with {self.names}"
            self.label_mapping = {label: i for i, label in enumerate(self.names)}


class LabelEncoder(Encoder):
    _config_class = LabelEncoderConfig
    config: LabelEncoderConfig
    output_feature_type = get_feature("ClassLabel")

    @sync_backup_config
    def set_params(self, **kwargs):
        self.config = self.config.replace_defaults(**kwargs)
        self.config.__post_init__()
        return self

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
    ) -> "LabelEncoder":
        self.config._input_columns = self._set_input_columns_and_arity(input_columns)
        return self._process_fit(
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

    def _get_features_out(self, X, **kwargs):
        features = super()._get_features_out(X, **kwargs)
        if features:
            input_names = self.config.feature_names_in_ or self.config.feature_idx_in_
            for name in input_names:
                label_feature = features.get(name)
                if isinstance(label_feature, get_feature("ClassLabel")):
                    label_names = label_feature.names
                    if self.config.names and isinstance(self.config.names[0], str):
                        self.config.label_mapping = {
                            label_names.index(label): i
                            for i, label in enumerate(self.config.names)
                        }

                label_feature = get_feature("ClassLabel")(names=self.config.names)
                features[name] = label_feature

        return features

    def _check_data(self, X):
        X_dims = DataHandler.get_shape(X)
        if len(X_dims) > 1:
            if X_dims[1] == 1:
                out = DataHandler.select_column(X, 0)
            else:
                raise ValueError(
                    f"Expected input to have 1 column, got {X_dims[1]} columns"
                )
        else:
            out = X
        return out

    def _process_fit_input(self, input, **kwargs):
        if self.config.names:
            kwargs["fn_kwargs"]["fn"] = None
        return super()._process_fit_input(input, **kwargs)

    def _fit_array(self, X):
        out = self._check_data(X)
        self.config.names = DataHandler.to_format(DataHandler.unique(out), "list")
        self.config._transform_process_desc = (
            f"Encoding labels with: {self.config.label_mapping}"
        )
        return self

    def _partial_fit_array(self, X):
        out = self._check_data(X)
        labs = DataHandler.to_format(DataHandler.unique(out), "list")
        self.config.names = np.unique(self.config.names + labs).tolist()
        return self

    def _pool_fit(self, fitted_processors):
        names = []
        for processor in fitted_processors:
            if isinstance(processor, LabelEncoder):
                names += processor.config.names
        self.config.names = np.unique(names).tolist()
        self.config.label_mapping = {
            label: i for i, label in enumerate(self.config.names)
        }
        self.config._transform_process_desc = (
            f"Encoding labels with: {self.config.label_mapping}"
        )
        return self

    def _transform_array(self, X):
        labs = self._check_data(X)
        labs = DataHandler.to_format(labs, "list")
        return [self.config.label_mapping.get(lab, -1) for lab in labs]
