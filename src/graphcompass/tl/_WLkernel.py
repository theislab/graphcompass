"""Functions for graph comparisons using the Weisfeiler-Lehman Graph kernel method."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm
from anndata import AnnData
from graphcompass.tl.utils import _calculate_graph, _get_igraph
from graphcompass.imports.wwl_package import pairwise_wasserstein_distance 


def compare_conditions(
    adata: AnnData,
    library_key: str = "sample",
    cluster_key: str = "cell_type",
    cell_type_key: str = None,
    compute_spatial_graphs: bool = True,
    num_iterations: int = 3,
    kwargs_nhood_enrich={},
    kwargs_spatial_neighbors={},
    copy: bool = False,
    **kwargs,
) -> AnnData:
    """
    Compares conditions based on entire spatial graphs using the WWL Kernel.

    Parameters
    ----------
    adata
        Annotated data object.
    library_key
        If multiple `library_id`, column in :attr:`anndata.AnnData.obs`
        which stores mapping between ``library_id`` and obs.
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where clustering is stored.
    cell_type_key
        Key in :attr:`anndata.AnnData.obs` where cell types are stored.
    compute_spatial_graphs
        Set to False if spatial graphs have been calculated or `sq.gr.spatial_neighbors` has already been run before.
    kwargs_nhood_enrich
        Additional arguments passed to :func:`squidpy.gr.nhood_enrichment` in `graphcompass.tl.utils._calculate_graph`.   
    kwargs_spatial_neighbors
        Additional arguments passed to :func:`squidpy.gr.spatial_neighbors` in `graphcompass.tl.utils._calculate_graph`. 
    kwargs
        Additional arguments passed to :func:`graphcompass.tl._calculate_graph`.
    copy
        Whether to return a copy of the Wasserstein distance object.
    """
    if compute_spatial_graphs:
        logging.info("Computing spatial graphs...")
        _calculate_graph(
                adata=adata,
                library_key=library_key,
                cluster_key=cluster_key,
                kwargs_nhood_enrich=kwargs_nhood_enrich,
                kwargs_spatial_neighbors=kwargs_spatial_neighbors,
                **kwargs
        )
    else:
        logging.info("Spatial graphs were previously computed. Skipping computing spatial graphs...")

    samples = adata.obs[library_key].unique()
    
    graphs = []
    node_features = []

    adata.uns["wl_kernel"] = {}
    if cell_type_key is not None:
        graphs = []
        adata.uns["wl_kernel"][cell_type_key] = {}
        for sample in samples:
            adata_sample = adata[adata.obs[library_key] == sample]
            g = _get_igraph(adata_sample, cluster_key=cell_type_key)
            graphs.append(g)
        
        wasserstein_distance = pairwise_wasserstein_distance(
            graphs,
            node_features=None,   # triggers categorical WL
            num_iterations=num_iterations
        )
        adata.uns["wl_kernel"][cell_type_key]["wasserstein_distance"] = pd.DataFrame(wasserstein_distance, columns=samples, index=samples)
                        
    else:
        logging.info("Defining node features...")
        for sample in tqdm(samples):
            adata_sample = adata[adata.obs[library_key] == sample]
            graphs.append(
                _get_igraph(
                    adata_sample, 
                    cluster_key=None
                )
            )
            features = adata_sample.X
            if isinstance(features, scipy.sparse._csr.csr_matrix):
                features = features.toarray()
            node_features.append(np.array(features))

        node_features = np.array(node_features, dtype=object)
        wasserstein_distance = pairwise_wasserstein_distance(
            graphs,
            node_features=node_features,  # triggers continuous WL
            num_iterations=num_iterations
        )
        adata.uns["wl_kernel"]["wasserstein_distance"] = pd.DataFrame(wasserstein_distance, columns=samples, index=samples)

    logging.info("Done!")
    if copy:
        return wasserstein_distance
