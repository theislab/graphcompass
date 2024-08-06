"""Functions for portrait- and diffusion-based graph comparisons."""

from __future__ import annotations

import scipy
import os
import netlsd
import tempfile

import itertools

import numpy as np
import pandas as pd

import networkx as nx

from tqdm import tqdm


from typing import (
    Optional, Union,  # noqa: F401
)

from joblib import delayed, Parallel


from numba import njit, prange  # noqa: F401

from anndata import AnnData

from scipy.stats import entropy

from squidpy._docs import d, inject_docs

from graphcompass.tl.utils import _calculate_graph
 

def compare_conditions(
    adata: AnnData,
    library_key: str = "sample",
    cluster_key: str = "cell_type",
    contrasts: list = None,
    cell_types: list = None,
    method: str = "portrait",
    portrait_flavour: Optional[str] = "python",
    max_depth: Optional[int] = 500,
    compute_spatial_graphs: bool = True,
    kwargs_nhood_enrich = {},
    kwargs_spatial_neighbors = {},
    copy: bool = False,
    **kwargs,
) -> AnnData:
    # 1. calculate graph 
    # 2. calculate graph distances
    # 3. calculate graph similarities
    # 4. calculate graph similarities for contrasts
    # in parallel manner

    """
    Compares conditions based on cell-type-specific subgraphs using distance methods.

    Parameters:
    ------------
    adata
        Annotated data object.
    library_key
        If multiple `library_id`, column in :attr:`anndata.AnnData.obs`
        which stores mapping between ``library_id`` and obs.
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where clustering is stored.
    contrasts
        List of tuples or lists defining which sample groups to compare.
    method
        Whether to use network portrait divergence (method = 'portrait') or diffusion (method = 'diffusion').
    portrait_flavour
        Whether to use the Python or C++ implementation of network portrait divergence when method is 'portrait'.
    max_depth
        Depth limit of the breadth-first search when method is 'portrait'.
    compute_spatial_graphs
        Set to False if spatial graphs have been calculated or `sq.gr.spatial_neighbors` has already been run before.
    kwargs_nhood_enrich
        Additional arguments passed to :func:`squidpy.gr.nhood_enrichment` in `graphcompass.tl.utils._calculate_graph`. 
    kwargs_spatial_neighbors
        Additional arguments passed to :func:`squidpy.gr.spatial_neighbors` in `graphcompass.tl.utils._calculate_graph`.
    kwargs
        Additional arguments passed to :func:`graphcompass.tl._calculate_graph`.
    copy
        Whether to return a copy of the pairwise similarities object.
    """

    if not isinstance(adata, AnnData):
        raise TypeError("Parameter 'adata' must be an AnnData object.")
    
    if copy:
        adata = adata.copy()

    # calculate graph
    
    if compute_spatial_graphs:
        print("Computing spatial graphs...")
        _calculate_graph(
            adata, 
            library_key=library_key,
            cluster_key=cluster_key, 
            kwargs_nhood_enrich=kwargs_nhood_enrich,
            kwargs_spatial_neighbors=kwargs_spatial_neighbors,
            **kwargs
        )
    else:
        print("Spatial graphs were previously computed. Skipping computing spatial graphs...")
    # calculate graph distances
    print("Computing graph similarities...")
    pairwise_similarities = _calculate_graph_distances(
        adata,
        library_key=library_key,
        cluster_key=cluster_key,
        cell_types=cell_types,
        method=method,
        portrait_flavour=portrait_flavour,
        max_depth=max_depth
    )
    
    # insert pairwise similarities into adata
    adata.uns["pairwise_similarities"] = pairwise_similarities

    print("Done!")
    if copy:
        return pairwise_similarities
    

def _calculate_graph_distances(
    adata: AnnData,
    library_key: str = "sample",
    cluster_key: str = "cell_type",
    cell_types: list = None,
    n_cell_types_per_graph: int = 1,
    method: str = "portrait",
    portrait_flavour: Optional[str] = "python",
    max_depth: Optional[int] = 500
) -> pd.DataFrame:
    
    if not isinstance(adata, AnnData):
        raise TypeError("Parameter 'adata' must be an AnnData object.")
    
    if cell_types is None:
        cell_types = adata.obs[cluster_key].unique().tolist()
        # TODO: add parameter to specify number of cell types allowed in a graph (default=1)
        # find all combinations of cell types with n = n_cell_types_per_graph
        # if n_cell_types_per_graph > 1:
        #     # add cell type combination
        # cell_types = itertools.product(cell_types, cell_types)

    if not isinstance(cell_types, list):
        raise TypeError("Parameter 'cell_types' must be a list.")
    
    samples = adata.obs[library_key].unique().tolist()
    

    # Pre-allocate memory for results
    results = []

    # Utilize all available CPU cores
    n_jobs = 4

    # Loop over cell types and calculate graph distances
    for cell_type in tqdm(cell_types):
        out = Parallel(n_jobs=n_jobs)(
            delayed(_calculate_graph_distance)(
                adata=adata, 
                sample_a=sample_a, 
                sample_b=sample_b, 
                cell_type=cell_type,
                library_key=library_key,
                cluster_key=cluster_key,
                method=method,
                portrait_flavour=portrait_flavour,
                max_depth=max_depth
            )
            for sample_a, sample_b in itertools.combinations(samples, 2)
        )
        # Collect results
        results.extend(out)

    # Create DataFrame from results
    column_names = ["sample_a", "sample_b", "cell_type", "ncells_a", "ncells_b", "density_a", "density_b", "similarity_score"]
    pairwise_similarities = pd.DataFrame(results, columns=column_names).dropna()

    return pairwise_similarities
    
    
def _calculate_graph_distance(
    adata: AnnData,
    sample_a: str,
    sample_b: str,
    cell_type: Union[str, list], # can be one cell type or list of cell types
    library_key: str,
    cluster_key: str,
    method: str,
    portrait_flavour: Optional[str] = "python",
    max_depth: Optional[int] = 500
) -> tuple[str, str, float, str]:
    """
    Calculates the similarity between two neighborhood graphs, taking into account graph emptiness and cell density.
    """
    if not isinstance(adata, AnnData):
        raise TypeError("Parameter 'adata' must be an AnnData object.")
    
    adata_sample_a = adata[adata.obs[library_key] == sample_a]
    adata_sample_b = adata[adata.obs[library_key] == sample_b]

    if isinstance(cell_type, str):
        adata_a = adata_sample_a[adata_sample_a.obs[cluster_key] == cell_type]
        adata_b = adata_sample_b[adata_sample_b.obs[cluster_key] == cell_type]
    elif isinstance(cell_type, list):
        adata_a = adata_sample_a[adata_sample_a.obs[cluster_key].isin(cell_type)]
        adata_b = adata_sample_b[adata_sample_b.obs[cluster_key].isin(cell_type)]
    else:
        raise ValueError(
            "Parameter 'cell_type' must be of type str or list."
        )
        
    ncells_a = len(adata_a)
    ncells_b = len(adata_b)

    graph_a = adata_a.obsp['spatial_connectivities']
    graph_b = adata_b.obsp['spatial_connectivities']

    # Integrate cell density into the comparison
    density_a = _calculate_graph_density(graph_a)
    density_b = _calculate_graph_density(graph_b)

    # Check for empty graphs
    if graph_a.size == 0 or graph_b.size == 0:
        # Handle empty graph case; e.g., assign maximum dissimilarity
        return (sample_a, sample_b, cell_type, ncells_a, ncells_b, density_a, density_b, 1.0 if graph_a.size != graph_b.size else 0.0)
    
    similarity_score = compare_graphs(graph_a, graph_b, method, portrait_flavour, max_depth)

    return (sample_a, sample_b, cell_type, ncells_a, ncells_b, density_a, density_b, similarity_score)


def _calculate_graph_density(graph: Union[nx.Graph, scipy.sparse._csr.csr_matrix]) -> float:
    """
    Calculates a density metric for a graph.
    """
    if isinstance(graph, scipy.sparse._csr.csr_matrix):
        graph = nx.from_scipy_sparse_array(graph)
    
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    
    if num_nodes > 1:
        # Example density calculation
        density = num_edges / (num_nodes * (num_nodes - 1))
    else:
        density = 0

    return density


def compare_graphs(graph_a: Union[nx.Graph, scipy.sparse._csr.csr_matrix], 
                   graph_b: Union[nx.Graph, scipy.sparse._csr.csr_matrix],
                   method: str,
                   portrait_flavour: Optional[str] = "python",
                   max_depth: Optional[int] = 500
) -> float:
    """
    Calculates the similarity between two neighborhood graphs.

    Parameters:
    ------------
    graph_a
        First squidpy-computed neighborhood graph.
    graph_b
        Second squidpy-computed neighborhood graph.
    method (str)
        Whether to use network portrait divergence (method = 'portrait') or diffusion (method = 'diffusion').
    portrait_flavour (str)
        Whether to use the Python or C++ implementation of network portrait divergence when method is 'portrait'.
    max_depth (int)
        Depth limit of the breadth-first search when method is 'portrait'.
    Returns
    -------
    If 'method' is portrait, a single float is returned. If graphs are identical, 0 is returned. If graphs are maximally different, 1 is returned.
    """
    if not method in ["diffusion", "portrait"]:
        raise ValueError("Parameter 'method' must be either 'diffusion' or 'portrait'.")

    if portrait_flavour not in ["python", "cpp"]:
        raise ValueError(
            "Parameter 'portrait_flavour' must be either 'python' (default) or 'cpp'."
        )

    if method == "portrait":
        if isinstance(graph_a, scipy.sparse._csr.csr_matrix):
            graph_a = nx.from_scipy_sparse_array(graph_a)

        if isinstance(graph_b, scipy.sparse._csr.csr_matrix):
            graph_b = nx.from_scipy_sparse_array(graph_b)

        similarity_score = _calculate_portrait_divergence(graph_a, graph_b, portrait_flavour, max_depth)

        return similarity_score

    elif method == "diffusion":
        descriptor_a = _diffusion_featurization(graph_a)
        descriptor_b = _diffusion_featurization(graph_b)

        similarity_score = netlsd.compare(descriptor_a, descriptor_b)

    return similarity_score


def compare_groups(
    pairwise_similarities: pd.DataFrame,
    sample_to_contrasts: pd.DataFrame,
    contrasts: list,
    output_format: str = "tidy"
) -> Union[pd.DataFrame, dict]:
    """
    Extracts the similarities between two contrasted groups of samples.

    Parameters:
    ------------
    pairwise_similarities
        pandas.DataFrame containing all pairwise similarity scores.
    sample_to_contrasts
        pandas.DataFrame establishing which sample_ids correspond to which contrast.
    contrasts
        List of tuples or lists defining which sample groups to compare.
    output_format
        Whether to return a tidy pandas.DataFrame or a dict.
    Returns
    -------
    All similarity scores for the defined contrasts.
    """

    if not isinstance(pairwise_similarities, pd.DataFrame):
        raise TypeError("Parameter 'pairwise_similarities' must be a pandas.DataFrame.")

    if not isinstance(sample_to_contrasts, pd.DataFrame):
        raise TypeError("Parameter 'sample_to_contrasts' must be a pandas.DataFrame.")

    if not all(c in ["sample_id", "contrast"] for c in sample_to_contrasts.columns):
        raise TypeError(
            "Parameter 'sample_to_contrasts' must have the two columns 'sample_id' and 'contrast'."
        )

    if not isinstance(contrasts, list):
        raise TypeError("Parameter 'contrasts' must be a list.")

    sample_ids = pairwise_similarities.columns

    if not all(
        [
            sample_id in sample_to_contrasts.sample_id.tolist()
            for sample_id in sample_ids
        ]
    ):
        raise ValueError(
            "All samples in 'pairwise_similarities' must also be found in 'sample_to_contrasts'."
        )

    if output_format not in ["tidy", "dict"]:
        raise ValueError("Parameter 'output_format' must be either 'tidy' or 'dict'.")

    if output_format == "tidy":
        output = pd.DataFrame()
    elif output_format == "dict":
        output = {}

    for contrast_a, contrast_b in contrasts:
        contrast_a_sample_ids = sample_to_contrasts.loc[
            sample_to_contrasts.contrast == contrast_a
        ].sample_id.tolist()
        contrast_b_sample_ids = sample_to_contrasts.loc[
            sample_to_contrasts.contrast == contrast_b
        ].sample_id.tolist()

        vals = pairwise_similarities.loc[
            contrast_a_sample_ids, contrast_b_sample_ids
        ].values
        vals = [item for sublist in vals for item in sublist]

        if output_format == "tidy":
            output = pd.concat(
                [
                    output,
                    pd.DataFrame(
                        {
                            "contrast": [f"{contrast_a} vs {contrast_b}"] * len(vals),
                            "vals": vals,
                        }
                    ),
                ]
            )

        elif output_format == "dict":
            output[f"{contrast_a} vs {contrast_b}"] = vals
    return output


def _diffusion_featurization(
    adjacency_matrix: Union[np.ndarray, scipy.sparse._csr.csr_matrix]
):
    """
    Computes a vector describing a graph using NetSLD (arXiv:1805.1071), given that graph's adjacency matrix.
    Parameters:
    ------------
    adjacency_matrix
        Array describing the graph in terms of the pairs of vertices that are adjacent.
        This array is stored in AnnData objects under .obsp['spatial_connectivities']
    Returns
    -------
        A vector representation of the graph.
    """
    descriptor = netlsd.heat(adjacency_matrix)
    return descriptor


@d.dedent
@inject_docs(tl="graphcompass.tl")
def _pad_portraits_to_same_size(
    B1: Union[np.ndarray, scipy.sparse._csr.csr_matrix],
    B2: Union[np.ndarray, scipy.sparse._csr.csr_matrix]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensures that two matrices are padded with zeros and/or trimmed of
    zeros to be the same shape.
    """
    ns, ms = B1.shape
    nl, ml = B2.shape

    # Bmats have N columns; find last *occupied* column and trim both down:
    lastcol1 = max(np.nonzero(B1)[1])
    lastcol2 = max(np.nonzero(B2)[1])
    lastcol = max(lastcol1, lastcol2)
    B1 = B1[:, : lastcol + 1]
    B2 = B2[:, : lastcol + 1]

    BigB1 = np.zeros((max(ns, nl), lastcol + 1))
    BigB2 = np.zeros((max(ns, nl), lastcol + 1))

    BigB1[: B1.shape[0], : B1.shape[1]] = B1
    BigB2[: B2.shape[0], : B2.shape[1]] = B2

    return BigB1, BigB2


def _graph_or_portrait(
        X: Union[nx.Graph, nx.DiGraph, scipy.sparse._csr.csr_matrix],
        portrait_flavour: str,
        max_depth: int
) -> Union[nx.Graph, nx.DiGraph, scipy.sparse._csr.csr_matrix]:
    """
    Checks if X is a nx (di)graph. Obtains its portrait if it is.
    Assumes it's a portrait otherwise and returns it.
    """
    if isinstance(X, (nx.Graph, nx.DiGraph)):
        return _calculate_portrait(X, portrait_flavour, max_depth)
    return X


def _calculate_portrait_divergence(
    G: Union[nx.Graph, scipy.sparse._csr.csr_matrix],
    H: Union[nx.Graph, scipy.sparse._csr.csr_matrix],
    portrait_flavour: str,
    max_depth: int
) -> float:
    """Computes the network portrait divergence between graphs G and H."""

    BG = _graph_or_portrait(G, portrait_flavour, max_depth)
    BH = _graph_or_portrait(H, portrait_flavour, max_depth)
    BG, BH = _pad_portraits_to_same_size(BG, BH)

    L, K = BG.shape
    V = np.tile(np.arange(K), (L, 1))

    XG = BG * V / (BG * V).sum()
    XH = BH * V / (BH * V).sum()

    # flatten distribution matrices as arrays:
    P = XG.ravel()
    Q = XH.ravel()

    # lastly, get JSD:
    M = 0.5 * (P + Q)
    KLDpm = entropy(P, M, base=2)
    KLDqm = entropy(Q, M, base=2)
    JSDpq = 0.5 * (KLDpm + KLDqm)

    return JSDpq


@d.dedent
@inject_docs(tl="graphcompass.tl")
def _calculate_portrait(
        graph: Union[nx.Graph, nx.DiGraph],
        portrait_flavour: str,
        max_depth: int,
        fname=None, 
        keepfile=False
) -> np.ndarray:

    """
    Computes and generates the portrait of a graph using the compiled B_matrix
    executable.

    Unoptimised; source: https://github.com/bagrow/network-portrait-divergence/blob/72993c368114c2e834142787579466d673232fa4/portrait_divergence.py

    """

    if portrait_flavour not in ["python", "cpp"]:
        raise ValueError("Parameter 'portrait_flavour' must be either 'python' or 'cpp'.")

    if portrait_flavour == "cpp":
        # file to save to:
        f = fname
        if fname is None:
            f = next(tempfile._get_candidate_names())

        # make sure nodes are 0,...,N-1 integers:
        graph = nx.convert_node_labels_to_integers(graph)

        # write edgelist:
        nx.write_edgelist(graph, f + ".edgelist", data=False)

        # make B-matrix:
        os.system("./B_matrix {}.edgelist {}.Bmat > /dev/null".format(f, f))
        portrait = np.loadtxt("{}.Bmat".format(f))

        # clean up:
        if not keepfile:
            os.remove(f + ".edgelist")
            os.remove(f + ".Bmat")

    elif portrait_flavour == "python":
        N = graph.number_of_nodes()
        # B indices are 0...dia x 0...N-1:
        B = np.zeros((max_depth + 1, N))

        max_path = 1
        adj = graph.adj
        # breadth-first search (BFS) for each node
        for starting_node in graph.nodes():
            nodes_visited = {starting_node: 0}
            search_queue = [starting_node]
            d = 1
            while search_queue and d <= max_depth:
                next_depth = []
                extend = next_depth.extend
                for n in search_queue:
                    l = [i for i in adj[n] if i not in nodes_visited]
                    extend(l)
                    for j in l:
                        nodes_visited[j] = d
                search_queue = next_depth
                d += 1

            node_distances = nodes_visited.values()
            max_node_distances = max(node_distances)

            curr_max_path = max_node_distances
            if curr_max_path > max_path:
                max_path = curr_max_path

            # build individual distribution:
            dict_distribution = dict.fromkeys(node_distances, 0)
            for d in node_distances:
                dict_distribution[d] += 1

            # add individual distribution to matrix:
            for shell, count in dict_distribution.items():
                B[shell][count] += 1

            # HACK: count starting nodes that have zero nodes in farther shells
            max_shell = max_depth
            while max_shell > max_node_distances:
                B[max_shell][0] += 1
                max_shell -= 1

        portrait = B[: max_path + 1, :]

    return portrait
