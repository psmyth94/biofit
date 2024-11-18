from dataclasses import dataclass, field
from pathlib import Path

from biocore import DataHandler
from biocore.utils.import_util import is_biosets_available

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedFeatureTypes
from biofit.visualization.plotting import BasePlotter, PlotterConfig


@dataclass
class ScalerPlotterConfig(PlotterConfig):
    processor_type: str = field(default="scaling", init=False, repr=False)
    r_source: str = field(
        default=(Path(__file__).parent / "plot_scaling.R").as_posix(),
        init=False,
        repr=False,
    )
    main_method: str = field(default="plot_scaler", init=False, repr=False)
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


class ScalerPlotter(BasePlotter):
    config_class = ScalerPlotterConfig
    config: ScalerPlotterConfig

    def plot_dataset(self, x1, x2, y1, y2):
        if is_biosets_available():
            from biosets import decode

            y1 = decode(y1) if y1 is not None else None
            y2 = decode(y2) if y2 is not None else None
        return self.plot_arrow(
            DataHandler.to_arrow(x1),
            DataHandler.to_arrow(x2),
            DataHandler.to_arrow(y1) if y1 is not None else None,
            DataHandler.to_arrow(y2) if y2 is not None else None,
        )

    def plot_arrow(self, x1, x2, y1, y2):
        return self.plotter(x1, x2, y1, y2, **self.config.get_params())
