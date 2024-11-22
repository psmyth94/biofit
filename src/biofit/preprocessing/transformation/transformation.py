from dataclasses import dataclass, field

from biofit.processing import BaseProcessor, ProcessorConfig


@dataclass
class TransformerConfig(ProcessorConfig):
    """Base class for Transformer processor configuration."""

    processor_type: str = field(default="transformation", init=False, repr=False)


class Transformer(BaseProcessor):
    """Base class for Transformer processors."""

    _config_class = TransformerConfig
    config: TransformerConfig
