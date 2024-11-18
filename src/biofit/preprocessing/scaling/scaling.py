from dataclasses import dataclass, field
from typing import List, Type

from biofit.integration.biosets import get_feature
from biofit.processing import BaseProcessor, ProcessorConfig


@dataclass
class ScalerConfig(ProcessorConfig):
    processor_type: str = field(default="scaling", init=False, repr=False)
    _fit_unused_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES")],
        init=False,
        repr=False,
    )
    _transform_unused_feature_types: List[Type] = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES")],
        init=False,
        repr=False,
    )


class Scaler(BaseProcessor):
    pass
