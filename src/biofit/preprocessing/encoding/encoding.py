from dataclasses import dataclass, field

from biofit.processing import BaseProcessor, ProcessorConfig


@dataclass
class EncoderConfig(ProcessorConfig):
    """Base class for feature extraction processor configurations."""

    processor_type: str = field(default="encoding", init=False, repr=False)


class Encoder(BaseProcessor):
    """Base class for feature extraction processors."""
