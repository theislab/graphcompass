from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import squidpy as sq
from anndata import AnnData
from matplotlib.axes import Axes

# plot graphs for specific cell types in specific library_key


def graphs_of_cells(
    adata: AnnData,
    library_key: str,
    cluster_key: str,
    samples: list[str] | str = None,
    cell_type: list[str] | str = None,
    connectivity_key: str = "spatial_connectivities",
    return_ax: bool = False,
    figsize: tuple[float, float] | None = (7, 30),
    dpi: int | None = 300,
    save: str | Path | None = None,
    **kwargs: Any,
) -> Axes | Sequence[Axes] | None:
    """
    Plot group comparison for each cell type.

    Parameters
    ----------
    adata
        Annotated data matrix.
    library_key
        Key in `adata.obs` where the library information is stored.
    cluster_key
        Key in `adata.obs` where the cluster information is stored.
    cell_type
        List of cell types to be plotted.
    regions
        List of regions to be plotted.
    fig
        Figure object to be used for plotting.
    ax
        Axes object to be used for plotting.
    return_ax
        If True, then return the axes object.
    figsize
        Figure size.
    dpi
        Figure resolution.
    save
        Filename under which to save the plot.
    **kwargs
        Keyword arguments to be passed to plotting functions.
    """

    if samples is None:
        samples = adata.obs[library_key].unique()
    if cell_type is None:
        cell_type = adata.obs[cluster_key].unique()

    ncols = 1
    nrows = len(samples)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)

    for i, sample in enumerate(samples):
        adata_sample = adata[adata.obs[library_key] == sample]
        adata_sample = adata_sample[adata_sample.obs[cluster_key].isin(cell_type)]

        sq.pl.spatial_scatter(
            adata_sample,
            library_key=library_key,
            library_id=sample,
            connectivity_key=connectivity_key,
            color=cluster_key,
            shape=None,
            ax=axs[i],
            size=10,
        )
        axs[i].set_title(sample)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=dpi)

    if return_ax:
        return axs
    else:
        plt.show()
