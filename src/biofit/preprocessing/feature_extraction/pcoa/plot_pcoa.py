from dataclasses import dataclass, field
from typing import Optional

from biofit.processing import SelectedColumnTypes
from biofit.utils.types import Unset
from biofit.visualization.plotting_utils import plot_dimension_reduction

from ..plot_feature_extraction import (
    FeatureExtractorPlotter,
    FeatureExtractorPlotterConfig,
)


@dataclass
class PCoAFeatureExtractorPlotterConfig(FeatureExtractorPlotterConfig):
    processor_name: str = field(default="pcoa", init=False, repr=False)
    title: str = "PCoA Plot"
    n_components: int = 3
    label_column: str = None
    group_column: str = None


class PCoAFeatureExtractorPlotter(FeatureExtractorPlotter):
    _config_class = PCoAFeatureExtractorPlotterConfig
    config: PCoAFeatureExtractorPlotterConfig

    def __init__(
        self,
        n_components: int = 3,
        label_column: str = None,
        group_column: str = None,
        config: Optional[PCoAFeatureExtractorPlotterConfig] = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            n_components=n_components,
            label_column=label_column,
            group_column=group_column,
            **kwargs,
        )

    def plot(
        self,
        X,
        labels=None,
        group=None,
        input_columns: SelectedColumnTypes = None,
        label_column: SelectedColumnTypes = None,
        group_column: SelectedColumnTypes = None,
        precomputed: bool = Unset("False"),
        pca_kwargs: dict = Unset("{}"),
        title: str = Unset("PCA Plot"),
        n_components: int = Unset(3),
        path=None,
        device: str = "png",
        fingerprint: str = None,
        unused_columns: SelectedColumnTypes = None,
        raise_if_missing: bool = True,
        show: bool = True,
    ):
        """Plot the PCoA plot.

        Args:
            X: The input data.
            labels: The labels for the data.
            group: The group for the data.
            input_columns: The input columns.
            label_column: The label column.
            group_column: The group column.
            precomputed: Whether the input data is precomputed.
            pcoa_kwargs: The additional kwargs for computing PCoA. Only used if precomputed is False.
            **kwargs: The additional kwargs.

        """
        return plot_dimension_reduction(
            X,
            labels=labels,
            group=group,
            input_columns=input_columns,
            label_column=label_column,
            group_column=group_column,
            method="pcoa" if not precomputed else None,
            method_kwargs=pca_kwargs,
            output_dir=path,
            device=device,
            fingerprint=fingerprint,
            unused_columns=unused_columns,
            raise_if_missing=raise_if_missing,
            show=show,
        )
