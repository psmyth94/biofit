from dataclasses import dataclass, field
from pathlib import Path

from biofit.visualization.plotting import BasePlotter, PlotterConfig


@dataclass
class FeatureSelectorPlotterConfig(PlotterConfig):
    processor_type: str = field(default="feature_selection", init=False, repr=False)
    r_source: str = field(
        default=(Path(__file__).parent / "plot_feature_selection.R").as_posix(),
        init=False,
        repr=False,
    )
    main_method: str = field(default="plot_feature_selector", init=False, repr=False)


class FeatureSelectorPlotter(BasePlotter):
    config_class = FeatureSelectorPlotterConfig
    config: FeatureSelectorPlotterConfig
