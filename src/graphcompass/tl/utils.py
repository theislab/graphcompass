"""Functions for graph calculation"""

from __future__ import annotations

import squidpy as sq

from typing import (
    Union,  # noqa: F401
)

import igraph
from joblib import delayed, Parallel


from numba import njit, prange  # noqa: F401

from anndata import AnnData


def _calculate_graph(
    adata: AnnData,
    library_key: str = "sample",
    cluster_key: str = "cell_type",
    copy: bool = False,
    kwargs_nhood_enrich={},
    kwargs_spatial_neighbors={},
):
    """
    Calculate the spatial graph and perform the sample-wise neighborhood enrichment analysis.

    Parameters:
    ------------
    adata
        Annotated data object.
    library_key
        If multiple `library_id`, column in :attr:`anndata.AnnData.obs`
        which stores mapping between ``library_id`` and obs.
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where clustering is stored.
    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the z-score and the enrichment count per sample.

    Otherwise, modifies the ``adata`` with the following keys:

        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_{library_key}_nhood_enrichment'][sample]['zscore']`` - the enrichment z-score.
        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_{library_key}_nhood_enrichment'][sample['count']`` - the enrichment count.
    """
    # calculate spatial neighbors for all samples
    sq.gr.spatial_neighbors(
        adata=adata, 
        library_key=library_key, 
        **kwargs_spatial_neighbors
    )
    nhood_enrichment = dict()
    # calculate neighborhood graphs per sample
    for sample in adata.obs[library_key].unique():
        adata_sample = adata[adata.obs[library_key] == sample].copy()
        sq.gr.nhood_enrichment(
            adata_sample, cluster_key=cluster_key, **kwargs_nhood_enrich
        )
        nhood_enrichment[sample] = adata_sample.uns[f"{cluster_key}_nhood_enrichment"]

    adata.uns[f"{cluster_key}_{library_key}_nhood_enrichment"] = nhood_enrichment

    if copy:
        return adata
    return



def _get_graph(
    adata,
    library_key: str,
    cluster_key: str,
    sample: str,
    cell_type: str,
):
    """
    Returns the sample-wise neighborhood enrichment analysis and sample-cell type graph connectivities

    Parameters:
    ------------
    adata
        Annotated data object.
    library_key
        If multiple `library_id`, column in :attr:`anndata.AnnData.obs`
        which stores mapping between ``library_id`` and obs.
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where clustering is stored.
    sample
        Sample ID of interest in library_key field
    cell_type
        Name of cell type of interest in cluster_key field
    Returns
    -------
    Dict of sample-wise zscore, counts and cell type spatial connectivities
    """
    adata_sample = adata[adata.obs[library_key] == sample].copy()

    nhood_enrichment = adata_sample.uns[
        f"{cluster_key}_{library_key}_nhood_enrichment"
    ][sample]

    adata_sample_ct = adata_sample[adata_sample.obs[cluster_key] == cell_type].copy()
    nhood_enrichment["spatial_connectivities"] = adata_sample_ct.obsp[
        "spatial_connectivities"
    ]

    return nhood_enrichment

def _get_igraph(
        adata, 
        connectivity_key: str ="spatial_connectivities",
        cluster_key: str = None,
    ) -> igraph.Graph:
    """
    Calculate iGraph object for a specific sample

    Parameters:
    ------------
    adata
        Annotated data object for specific sample.
    connectivity_key
        key for adjacency matrix
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where clustering is stored. 
    Returns
    -------
    iGraph object 
    """
    
    A = adata.obsp[connectivity_key].toarray()

    g = igraph.Graph.Adjacency((A > 0).tolist())

    g.es['weight'] = A[A.nonzero()]
    if cluster_key:
        g.vs['label'] = adata.obs[cluster_key] 

    return g