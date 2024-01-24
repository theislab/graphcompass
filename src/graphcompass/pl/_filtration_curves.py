import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from anndata import AnnData

import matplotlib
from matplotlib.axes import Axes

from typing import Any, Sequence, Tuple, Union


def compare_conditions(
    adata: AnnData,
    node_labels: set,
    metric_key: str = "filtration_curves",
    return_ax: bool = False,
    figsize: Union[Tuple[float, float], None] = None,
    dpi: Union[int, None] = 300,
    palette: str = "Set2",
    save: Union[str, Path, None] = None,
    **kwargs: Any,
) -> Union[Axes, Sequence[Axes], None]:
    """
    Plot group comparison for full samples.

    Parameters
    ----------
    adata
        Annotated data matrix.
    node_labels
        Set of node labels.
    metric_key
        Key in `adata.uns` where the metric of interest is stored.
    return_ax
        If True, then return the axes object.
    figsize
        Figure size.
    dpi
        Figure resolution.
    palette
        matplotlib colormap name.
    save
        Filename under which to save the plot.
    **kwargs
        Keyword arguments to be passed to plotting functions.
    """
    filtration_curves = adata.uns[metric_key]["curves"]
    threshold_vals = adata.uns[metric_key]["threshold_vals"]

    n_node_labels = len(node_labels)

    plt.rcParams["font.size"] = 12

    # Create subplots
    if figsize is not None:
        fig, axes = plt.subplots(
            nrows=1, ncols=n_node_labels, figsize=figsize
        )
    else:
        fig, axes = plt.subplots(
            nrows=1, ncols=n_node_labels, figsize=(8 * n_node_labels, 6)
        )

    # Colormap for categorical values in column 'graph_label'
    cmap = matplotlib.colormaps[palette]

    # Get unique categories in 'graph_label'
    unique_categories = np.unique(
        np.concatenate(
            [df["graph_label"].unique() for df in filtration_curves]
        )
    )

    # Create a color map dictionary for each unique category
    category_color_map = {
        category: cmap(i) for i, category in enumerate(unique_categories)
    }

    # Iterate over each DataFrame and cell type
    for df in filtration_curves:
        label = pd.unique(df.graph_label)
        assert label.size == 1
        for i, key in enumerate(node_labels):
            try:
                axes[i].step(
                    df.weight,
                    df[key],
                    where='pre',
                    label=label[0],
                    color=category_color_map[label[0]],
                    alpha=0.15
                )
            except KeyError:
                continue

    # Create mean filtration curves
    combined_df = pd.concat(filtration_curves)
    grouped_df = combined_df.groupby(['graph_label', 'weight'])
    average_df = grouped_df.mean().reset_index()

    custom_xticks = [round(val, 1) for val in threshold_vals]

    for i, key in enumerate(node_labels):
        # Plot mean filtration curves
        for graph_label in pd.unique(average_df.graph_label):
            df = average_df[average_df.graph_label == graph_label]
            axes[i].step(
                df.weight,
                df[key],
                where="pre",
                label=graph_label,
                color=category_color_map[graph_label]
            )

        # Set custom values on the x-axis
        axes[i].set_xticks(custom_xticks)

        # Set title
        axes[i].set_title(f'Step Plot for Column {key}')

        # Add legend
        handles, labels = axes[i].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[i].legend(by_label.values(), by_label.keys(), title='Graph label')

    plt.tight_layout()
    if save is not None:
        plt.savefig(save, dpi=dpi)
    if return_ax:
        return axes
    else:
        plt.show()
