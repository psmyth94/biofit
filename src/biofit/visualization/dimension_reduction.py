import json
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow as pa
from biocore import DataHandler
from biocore.utils.import_util import requires_backends

from biofit.integration.biosets import get_feature
from biofit.processing import SelectedColumnTypes, SelectedFeatureTypes
from biofit.utils.types import Unset
from biofit.visualization.plotting import BasePlotter, PlotterConfig
from biofit.visualization.plotting_utils import get_distinct_colors


@dataclass
class DimensionReducerPlotterConfig(PlotterConfig):
    processor_type: str = field(default="feature_extractor", init=False, repr=False)
    _fit_input_feature_types: SelectedFeatureTypes = field(
        default_factory=lambda: [
            None,
            get_feature("TARGET_FEATURE_TYPES"),
            get_feature("METADATA_FEATURE_TYPES"),
        ],
        init=False,
        repr=False,
    )
    _transform_input_feature_types: SelectedFeatureTypes = field(
        default_factory=lambda: [
            None,
            get_feature("TARGET_FEATURE_TYPES"),
            get_feature("METADATA_FEATURE_TYPES"),
        ],
        init=False,
        repr=False,
    )
    _fit_unused_feature_types: SelectedFeatureTypes = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES"), None, None],
        init=False,
        repr=False,
    )
    _transform_unused_feature_types: SelectedFeatureTypes = field(
        default_factory=lambda: [get_feature("METADATA_FEATURE_TYPES"), None, None],
        init=False,
        repr=False,
    )

    title: str = "Dimension Reduction Plot"
    colormap: str = "nipy_spectral"
    n_components: int = 3
    label_column: str = None
    group_column: str = None


class DimensionReductionPlotter(BasePlotter):
    """Base class for feature extraction processors."""

    _config_class = DimensionReducerPlotterConfig
    config: DimensionReducerPlotterConfig

    def __init__(
        self,
        label_column: str = None,
        group_column: str = None,
        title: str = Unset('"Dimension Reduction Plot"'),
        colormap: str = Unset('"nipy_spectral"'),
        n_components: int = Unset("3"),
        config: DimensionReducerPlotterConfig = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            n_components=n_components,
            label_column=label_column,
            group_column=group_column,
            title=title,
            colormap=colormap,
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
        title: str = Unset('"Dimension Reduction Plot"'),
        colormap: str = Unset('"nipy_spectral"'),
        n_components: int = Unset("3"),
        dimension_reducer=None,
        path: str = None,
        device: str = "png",
        fingerprint: str = None,
        unused_columns: SelectedColumnTypes = None,
        raise_if_missing: bool = True,
        show=True,
    ):
        """Plot the PCoA plot.

        Args:
            X (array_like):
                The input data. Can be a Dataset, polars/pandas DataFrame, numpy array, or Arrow table.
            labels (array_like, *optional*):
                The labels for the data. Dictates the color of the points in the plot.
                Must be an array of values with the same length as the number of rows in X.
                If not provided, the points will be colored by group.
            group (array_like, *optional*):
                The group for the data. Dictates the shape of the points in the plot.
                Must be an array of values with the same length as the number of rows in X.
            pcoa (array_like, *optional*):
                The fitted PCoAFeatureExtractor object. Used to extract the eigvals for the explained variance ratio.
                If not provided, the explained variance ratio will not be displayed.
            **kwargs:
                Additional keyword arguments to pass to the plot:
                    n_components (int, 2):
                        The number of components to plot. Defaults to 2.
                    label_name (str, *optional*):
                        The name of the labels. Used for the legend and retrieving the values from X if labels is not provided.
                    group_name (str, *optional*):
                        The name of the group. Used for the legend and retrieving the values from X if group is not provided.

        """
        self.config._input_columns = self._set_input_columns_and_arity(
            input_columns, label_column, group_column
        )
        return self._plot(
            X,
            labels,
            group,
            n_components=n_components,
            dimension_reducer=dimension_reducer,
            path=path,
            device=device,
            fingerprint=fingerprint,
            unused_columns=unused_columns,
            raise_if_missing=raise_if_missing,
            show=show,
        )

    def plot_dataset(self, X, labels=None, group=None, dimension_reducer=None):
        from biosets import decode

        if labels is not None:
            labels = decode(labels)

        return self.plot_arrow(
            DataHandler.to_arrow(X),
            DataHandler.to_arrow(labels) if labels is not None else None,
            DataHandler.to_arrow(group) if group is not None else None,
            dimension_reducer,
        )

    def plot_arrow(self, X, labels=None, group=None, dimension_reducer=None):
        def get_int2str_converter(x) -> np.ndarray:
            if not isinstance(x, pa.Table):
                return lambda x: x

            def get_features(x: pa.Table):
                metadata = x.schema.metadata
                if metadata:
                    if b"huggingface" in metadata:
                        metadata = json.loads(metadata[b"huggingface"].decode("utf-8"))

                    if "info" in metadata and "features" in metadata["info"]:
                        return metadata["info"]["features"]

                return {}

            metadata = get_features(x)
            lab_col = x.column_names[0]
            if metadata and lab_col in metadata:
                label_metadata = metadata[lab_col]
                if label_metadata["_type"] == "ClassLabel":
                    label_metadata.pop("_type")
                    cls_label = get_feature("ClassLabel")(**label_metadata)
                    return cls_label.int2str
            return lambda x: x

        def pairplot(
            tbl: pd.DataFrame,
            n_components,
            ex_var_ratio=None,
            label_name=None,
            group_name=None,
            marker_dict=None,
            color_dict=None,
        ):
            requires_backends(pairplot, "seaborn")
            requires_backends(pairplot, "matplotlib")
            import matplotlib.pyplot as plt
            import seaborn as sns
            from matplotlib.lines import Line2D

            features = tbl.columns[:n_components]
            fig, axes = plt.subplots(n_components, n_components, figsize=(15, 15))

            for i, f1 in enumerate(features):
                for j, f2 in enumerate(features):
                    ax = axes[i, j]
                    if i != j:
                        if label_name is not None and group_name is not None:
                            for (lab, grp), df_group in tbl.groupby(
                                [
                                    label_name,
                                    group_name,
                                ]
                            ):
                                sns.scatterplot(
                                    x=f2,
                                    y=f1,
                                    data=df_group,
                                    ax=ax,
                                    color=color_dict[lab],
                                    marker=marker_dict[grp],
                                )
                        elif label_name is not None:
                            for lab, df_lab in tbl.groupby(label_name):
                                sns.scatterplot(
                                    x=f2,
                                    y=f1,
                                    data=df_lab,
                                    ax=ax,
                                    color=color_dict[lab],
                                )
                        elif group_name is not None:
                            for grp, df_group in tbl.groupby(group_name):
                                sns.scatterplot(
                                    x=f2,
                                    y=f1,
                                    data=df_group,
                                    ax=ax,
                                    marker=marker_dict[grp],
                                )
                        else:
                            sns.scatterplot(x=f2, y=f1, data=tbl, ax=ax)

                    else:
                        if label_name is not None:
                            for lab, df_lab in tbl.groupby(label_name):
                                sns.kdeplot(
                                    df_lab[f1],
                                    ax=ax,
                                    color=color_dict[lab],
                                    fill=True,
                                )
                        else:
                            sns.kdeplot(tbl[f1], ax=ax, fill=True)
                        if ex_var_ratio is not None:
                            plt.text(
                                0.1,
                                0.9,
                                f"Dim{i + 1}: {float(ex_var_ratio[i]) * 100:.2f}%",
                                transform=ax.transAxes,
                            )
                    # Set only left and bottom borders
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

                    # Enable ticks on left and bottom only
                    ax.xaxis.set_tick_params(which="both", bottom=True)
                    ax.yaxis.set_tick_params(which="both", left=True)

                    # Set axis labels
                    if i == n_components - 1:
                        ax.set_xlabel(f2)
                    else:
                        ax.set_xlabel("")
                        ax.set_xticklabels([])

                    if j == 0:
                        ax.set_ylabel(f1)
                    else:
                        ax.set_ylabel("")
                        ax.set_yticklabels([])

            if label_name:
                label_lines = [
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        marker="o",
                        linestyle="",
                        markersize=10,
                    )
                    for color in color_dict.values()
                ]
                u_labels = list(color_dict.keys())
                n_labels = len(u_labels)
                label_height = 0.5 + 0.216 / 15 * (n_labels + 1) / 2 + 0.01
                label_legend = fig.legend(
                    label_lines,
                    u_labels,
                    title=label_name,
                    loc="center left",
                    bbox_to_anchor=(0.90, label_height),
                )

            if group_name:
                group_lines = [
                    Line2D(
                        [0],
                        [0],
                        color="gray",
                        marker=marker,
                        linestyle="",
                        markersize=10,
                    )
                    for marker in marker_dict.values()
                ]
                u_group = list(marker_dict.keys())
                n_group = len(u_group)

                group_height = 0.5 - 0.216 / 15 * (n_group + 1) / 2 - 0.01
                fig.legend(
                    group_lines,
                    list(marker_dict.keys()),
                    title=group_name,
                    loc="center left",
                    bbox_to_anchor=(0.90, group_height),
                )
                if label_name:
                    fig.add_artist(label_legend)

            if self.config.title:
                fig.suptitle(self.config.title, fontsize=16)

            return fig

        warnings.filterwarnings("ignore")
        n_components = self.config.n_components
        tbl = DataHandler.to_format(X, "pandas").iloc[:, :n_components]
        labs = None
        label_name = self.config.label_column or "labels"
        u_labels = None
        grp = None
        grp_name = self.config.group_column or None
        u_group = None
        if labels is not None:
            labs = DataHandler.to_format(labels, "pandas")
            if isinstance(labs, pd.DataFrame):
                label_name = labs.columns[0] if label_name is None else label_name
                labs = labs.iloc[:, 0]
            elif isinstance(labs, pd.Series):
                label_name = labs.name if label_name is None else label_name
            labs = labs.fillna("None")
            u_labels = np.unique(labs)
        if group is not None:
            grp = DataHandler.to_format(group, "pandas")
            if isinstance(grp, pd.DataFrame):
                grp_name = grp.columns[0] if grp_name is None else grp_name
                grp = grp.iloc[:, 0]
            elif isinstance(grp, pd.Series):
                grp_name = grp.name if grp_name is None else grp_name
            u_group = np.unique(grp)

        if labs is not None:
            converter = get_int2str_converter(labels)
            dtype = DataHandler.get_dtypes(labels)
            if "int" in next(iter(dtype.values())):
                encodings = labs.to_numpy()
                encodings_dims = encodings.shape
                if len(encodings_dims) == 1:
                    labs = converter(encodings)
                elif encodings_dims[1] == 1:
                    labs = converter(encodings[:, 0])

        if labs is not None:
            u_labels = np.unique(labs)

        ex_var_ratio = None
        if dimension_reducer is not None:
            if hasattr(dimension_reducer.config, "eigvals"):
                eigvals = dimension_reducer.config.eigvals
                ex_var_ratio = eigvals / eigvals.sum()
            elif hasattr(dimension_reducer.config, "estimator"):
                ex_var_ratio = (
                    dimension_reducer.config.estimator.explained_variance_ratio_
                )

        tbl.columns = [f"Dim{i + 1}" for i in range(len(tbl.columns))]

        marker_styles = ["o", "v", "^", "<", ">", "s", "p", "*", "H", "X"]
        if u_group is not None:
            marker_dict = {
                label: marker_styles[i % len(marker_styles)]
                for i, label in enumerate(u_group)
            }
        else:
            marker_dict = None

        if labs is not None and grp is not None:
            tbl = tbl.assign(**{label_name: labs, grp_name: grp})
        elif labs is not None:
            tbl = tbl.assign(**{label_name: labs})
        elif grp is not None:
            tbl = tbl.assign(**{grp_name: grp})

        if labs is not None:
            color_dict = {
                label: color
                for label, color in zip(
                    u_labels, get_distinct_colors(len(u_labels), self.config.colormap)
                )
            }
        elif grp is not None:
            color_dict = {
                label: color
                for label, color in zip(
                    u_group, get_distinct_colors(len(u_group), self.config.colormap)
                )
            }
        else:
            color_dict = None

        fig = pairplot(
            tbl,
            n_components,
            ex_var_ratio,
            label_name,
            grp_name,
            marker_dict,
            color_dict,
        )

        warnings.filterwarnings("default")

        # save the figure to the path
        if self.config.path:
            fig.savefig(self.config.path)
        return fig
