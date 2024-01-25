
import itertools
from tqdm import tqdm

import numpy as np
import pandas as pd

import scipy.sparse

from scipy.spatial.distance import euclidean
from scipy.sparse import csr_matrix

from anndata import AnnData
from igraph import Graph

from typing import Optional

from graphcompass.tl.utils import _calculate_graph


def compare_conditions(
    adata: AnnData,
    library_key: str = "sample",
    cluster_key: str = "cell_type",
    condition_key: str = "condition",
    attribute: str = "weight",
    sample_ids: Optional[list[str]] = None,
    compute_spatial_graphs: bool = True,
    kwargs_nhood_enrich: dict = {},
    kwargs_spatial_neighbors: dict = {},
    copy: bool = False,
    **kwargs,
) -> AnnData:
    """
    Compare conditions based on filteration curves.

    Parameters:
    ------------
    adata
        Annotated data object.
    library_key
        If multiple `library_id`, column in :attr:`anndata.AnnData.obs`
        which stores mapping between ``library_id`` and obs.
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where clustering is stored.
    condition_key
        Key in :attr:`anndata.AnnData.obs` where condition is stored.
    attribute
        Edge attribute name.
    sample_ids
        List of sample/library identifiers.
    compute_spatial_graphs
        Set False if spatial graphs has been calculated or
        `sq.gr.spatial_neighbors` has already been run before.
    kwargs_nhood_enrich
        Additional arguments passed to :func:`squidpy.gr.nhood_enrichment` in
        `graphcompass.tl.utils._calculate_graph`.
    kwargs_spatial_neighbors
        Additional arguments passed to :func:`squidpy.gr.spatial_neighbors` in
        `graphcompass.tl.utils._calculate_graph`.
    kwargs
        Additional arguments passed to :func:`graphcompass.tl.utils._calculate_graph`.
    copy
        Whether to copy the AnnData object or modify it in place.
    Returns
    -------
    If ``copy = True``, returns a :class:`list` of filtration dataframes.
    """
    if not isinstance(adata, AnnData):
        raise TypeError("Parameter 'adata' must be an AnnData object.")

    if copy:
        adata = adata.copy()

    # Create graph from spatial coordinates
    print("Computing spatial graph...")
    if compute_spatial_graphs:
        _calculate_graph(
            adata,
            library_key=library_key,
            cluster_key=cluster_key,
            kwargs_nhood_enrich=kwargs_nhood_enrich,
            kwargs_spatial_neighbors=kwargs_spatial_neighbors,
            **kwargs
        )

    print("Computing edge weights...")
    # Compute edge weights
    edge_weights = _compute_edge_weights(
        gene_expression_matrix=adata.X,
        adjacency_matrix=adata.obsp['spatial_connectivities']
    )

    adata.obsp['edge_weights'] = edge_weights

    graphs = []
    node_labels = set()

    if sample_ids is not None:
        samples = sample_ids
    else:
        samples = adata.obs[library_key].unique()

    for sample in tqdm(samples):
        # Create an igraph graph (initially unweighted) from the adjacency
        # matrix
        patient_data = adata[adata.obs[library_key] == sample]
        adj_matrix = patient_data.obsp['edge_weights']
        graph = Graph.Adjacency((adj_matrix > 0), mode='undirected')

        # Extract weights from the sparse matrix and assign them to the edges
        if scipy.sparse.issparse(adj_matrix):
            weights = adj_matrix.tocoo()
        else:
            weights = adj_matrix

        for i, j, weight in zip(weights.row, weights.col, weights.data):
            if i <= j:  # This ensures that each edge is considered only once
                edge_id = graph.get_eid(i, j)
                graph.es[edge_id]['weight'] = weight

        # Assign cell type labels to nodes
        labels = patient_data.obs[cluster_key]

        for i, attribute in enumerate(labels):
            graph.vs[i]['label'] = attribute

        # Assign label to graph
        graph_label = pd.unique(patient_data.obs[condition_key])
        assert graph_label.size == 1
        graph['label'] = graph_label[0]

        graphs.append(graph)
        node_labels.update(set(labels))

    print("Computing edge weight threshold values...")
    # Determine edge weight threshold values
    edge_weights = np.array([])

    for g in graphs:
        edge_weights = np.append(edge_weights, g.es['weight'])

    # Sort the edge weight array
    sorted_array = np.sort(edge_weights)

    # To calculate 10 thresholds, create an array of percentiles from 0 to
    # 100 with 10 steps
    percentiles = np.linspace(0, 100, 11)
    thresholds = np.percentile(sorted_array, percentiles)

    # The 'thresholds' array now contains the 10 thresholds
    threshold_vals = thresholds[1:]

    print("Creating filtration curves...")
    filtration_curves = _create_filtration_curves(
        graphs,
        threshold_vals=threshold_vals
    )

    print("Done!")
    adata.uns["filtration_curves"] = {}
    adata.uns["filtration_curves"]["curves"] = filtration_curves
    adata.uns["filtration_curves"]["threshold_vals"] = threshold_vals

    if copy:
        return filtration_curves


def _compute_edge_weights(gene_expression_matrix, adjacency_matrix):
    """
    Computes edge weights based on the Euclidean distance between the gene
    expression matrices of two neighboring nodes.

    Parameters:
    ------------
    gene_expression_matrix:
        Gene expression data, typically stored in adata.X
    adjacency_matrix:
        Connection matrix built by Squidpy's spatial_neighbors function

    Returns
    -------
        Edge weights.
    """
    edge_weights = csr_matrix(adjacency_matrix.shape, dtype=float)

    rows, cols = adjacency_matrix.nonzero()
    for i, j in zip(rows, cols):
        if i < j:  # To avoid duplicate computation for undirected graphs
            try:
                distance = euclidean(
                    gene_expression_matrix[i],
                    gene_expression_matrix[j],
                )
            except ValueError:
                distance = euclidean(
                    gene_expression_matrix[i].toarray()[0],
                    gene_expression_matrix[j].toarray()[0],
                )
            edge_weights[i, j] = distance
            edge_weights[j, i] = distance  # Assuming undirected graph

    return edge_weights


def _create_filtration_curves(
        graphs,
        threshold_vals,
):
    """
    Creates the node label filtration curves.

    Given a dataset of igraph graphs, we create a filtration curve using node
    label histograms. Given an edge weight threshold, we generate a node label
    histogram for the subgraph induced by all edges with weight less than or
    equal to that given threshold. This function is based on code for the KDD
    2021 paper 'Filtration Curves for Graph Representation':
        GitHub: https://github.com/BorgwardtLab/filtration_curves
        Paper: https://doi.org/10.1145/3447548.3467442

    Parameters:
    ------------
    graphs: list
        A collection of graphs
    threshold vals: list
        List of edge weight threshold values

    Returns
    -------
        List of filtrations.
    """
    # Ensures edge weights were assigned
    for graph in graphs:
        assert "weight" in graph.edge_attributes()

    # Get all potential node labels to make sure that the distribution
    # can be calculated correctly later on.
    node_labels = sorted(set(
        itertools.chain.from_iterable(graph.vs['label'] for graph in graphs)
    ))

    label_to_index = {
        label: index for index, label in enumerate(sorted(node_labels))
    }

    # Builds the filtration using the edge weights
    filtrated_graphs = [
        _filtration_by_edge_attribute(
            graph,
            threshold_vals,
            attribute='weight',
            delete_nodes=True,
            stop_early=True
        )
        for graph in tqdm(graphs)
    ]

    # Create a data frame for every graph and store it; the output is
    # determined by the input filename, albeit with a new extension.

    list_of_df = []

    for index, filtrated_graph in enumerate(tqdm(filtrated_graphs)):

        columns = ['graph_label', 'weight'] + node_labels
        rows = []

        distributions = _node_label_distribution(
            filtrated_graph,
            label_to_index
        )

        for weight, counts in distributions:
            row = {
                'graph_label': graphs[index]['label'],
                'weight': weight
            }
            row.update(
                {
                    str(node_label): count
                    for node_label, count in zip(node_labels, counts)
                }
            )
            rows.append(row)

        df = pd.DataFrame(rows, columns=columns)
        list_of_df.append(df)

    return list_of_df


def _node_label_distribution(filtration, label_to_index):
    """
    Calculates the node label distribution along a filtration.

    Given a filtration from an individual graph, we calculate the node label
    histogram (i.e. the count of each unique label) at each step along that
    filtration, and return a list of the edge weight threshold and its
    associated count vector. This function is based on code for the KDD 2021
    paper 'Filtration Curves for Graph Representation':
        GitHub: https://github.com/BorgwardtLab/filtration_curves
        Paper: https://doi.org/10.1145/3447548.3467442

    Parameters:
    ------------
    filtration : list
        A filtration of graphs
    label_to_index : mappable
        A map between labels and indices, required to calculate the histogram

    Returns
    -------
    Label distributions along the filtration. Each entry is a tuple
    consisting of the weight threshold followed by a count vector.
    """
    # D will contain the distributions as count vectors; this distribution is
    # calculated for every step of the filtration.
    D = []

    for weight, graph in filtration:
        labels = graph.vs['label']
        counts = np.zeros(len(label_to_index))

        for label in labels:
            index = label_to_index[label]
            counts[index] += 1

        # The conversion ensures that we can arrange everything later on in a
        # 'pd.series'.
        D.append((weight, counts.tolist()))

    return D


def _filtration_by_edge_attribute(
        graph,
        threshold_vals,
        attribute='weight',
        delete_nodes=False,
        stop_early=False
):
    """
    Calculates a filtration of a graph based on an edge attribute of the graph.
    This function is based on code for the KDD 2021 paper 'Filtration Curves
    for Graph Representation':
        GitHub: https://github.com/BorgwardtLab/filtration_curves
        Paper: https://doi.org/10.1145/3447548.3467442

    Parameters:
    ------------
    graph
        igraph Graph
    threshold_vals
        Edge weight thresholds
    attribute
        Edge attribute name
    delete_nodes
        If set, removes nodes from the filtration if none of their incident
        edges is part of the subgraph. By default, all nodes are kept
    stop_early
        If set, stops the filtration as soon as the number of nodes has been
        reached

    Returns
    -------
    Filtration as a list of tuples, where each tuple consists of the weight
    threshold and the graph.
    """

    weights = graph.es[attribute]
    weights = np.array(weights)

    # Represents the filtration of graphs according to the client-specified
    # attribute.
    F = []

    n_nodes = graph.vcount()

    # Checks if graphs have more than a single edge.
    if weights.size != 1:
        weights = weights
        x = False
    else:
        weights = np.array([[weights]])
        x = True

    for weight in sorted(np.unique(threshold_vals)):

        if x:
            weight = weight[0]
        edges = graph.es.select(lambda edge: edge[attribute] <= weight)
        subgraph = edges.subgraph(delete_vertices=delete_nodes)

        # Store weight and the subgraph induced by the selected edges as one
        # part of the filtration. The client can decide whether each node that
        # is not adjacent to any edge should be removed from the filtration.
        F.append((weight, subgraph))

        # If we have assembled enough nodes already, we do not need to
        # continue.
        if stop_early and subgraph.vcount() == n_nodes:
            break

    return F
