from dataclasses import dataclass, field

from biofit.integration.biosets import get_feature
from biofit.processing import BaseProcessor, ProcessorConfig, SelectedFeatureTypes


@dataclass
class FeatureExtractorConfig(ProcessorConfig):
    """Base class for feature extraction processor configurations."""

    processor_type: str = field(default="feature_extraction", init=False, repr=False)
    _fit_unused_feature_types: SelectedFeatureTypes = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES")],
        init=False,
        repr=False,
    )
    _transform_unused_feature_types: SelectedFeatureTypes = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES")],
        init=False,
        repr=False,
    )


class FeatureExtractor(BaseProcessor):
    """Base class for feature extraction processors."""
