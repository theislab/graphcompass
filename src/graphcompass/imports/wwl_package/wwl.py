######## This file is adapted from https://github.com/BorgwardtLab/WWL/blob/master/src/wwl/wwl.py ########

import sys
import logging

import torch
from geomloss import SamplesLoss
from sklearn.metrics.pairwise import laplacian_kernel

from .propagation_scheme import WeisfeilerLehman, ContinuousWeisfeilerLehman

logging.basicConfig(level=logging.INFO)

def logging_config(level='DEBUG'):
    level = logging.getLevelName(level.upper())
    logging.basicConfig(level=level)
    pass

def _compute_wasserstein_distance_geomloss(label_sequences, categorical=False, blur=0.05, p=2):
    """
    Compute pairwise Wasserstein distances between graph node embeddings using GeomLoss.
    Automatically uses GPU if available.

    Args:
        label_sequences: list of arrays (each array is [n_nodes, d] for a graph)
        categorical: if True, assumes discrete labels; else assumes continuous node embeddings
        blur: smoothing parameter for Sinkhorn (smaller = closer to EMD)
        p: power for cost (usually 2 for Euclidean squared)

    Returns:
        Distance matrix (n_graphs x n_graphs)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sinkhorn = SamplesLoss("sinkhorn", p=p, blur=blur)

    n = len(label_sequences)
    M = torch.zeros((n, n), device=device)

    for i, emb_i in enumerate(label_sequences):
        x_i = torch.tensor(emb_i, dtype=torch.float32, device=device)

        for j in range(i, n):
            x_j = torch.tensor(label_sequences[j], dtype=torch.float32, device=device)

            # Uniform weights
            a = torch.ones(x_i.shape[0], device=device) / x_i.shape[0]
            b = torch.ones(x_j.shape[0], device=device) / x_j.shape[0]

            dist = sinkhorn(a, x_i, b, x_j)
            M[i, j] = dist
            M[j, i] = dist  # symmetric

    return M.cpu().numpy()

def pairwise_wasserstein_distance(X, node_features = None, num_iterations=3, sinkhorn=False, enforce_continuous=False):
    """
    Pairwise computation of the Wasserstein distance between embeddings of the 
    graphs in X.
    args:
        X (List[ig.graphs]): List of graphs
        node_features (array): Array containing the node features for continuously attributed graphs
        num_iterations (int): Number of iterations for the propagation scheme
        sinkhorn (bool): Indicates whether sinkhorn approximation should be used
    """
    # First check if the graphs are continuous vs categorical
    categorical = True
    if enforce_continuous:
        logging.info('Enforce continous flag is on, using CONTINUOUS propagation scheme.')
        categorical = False
    elif node_features is not None:
        logging.info('Continuous node features provided, using CONTINUOUS propagation scheme.')
        categorical = False
    else:
        for g in X:
            if not 'label' in g.vs.attribute_names():
                logging.info('No label attributed to graphs, use degree instead and use CONTINUOUS propagation scheme.')
                categorical = False
                break
        if categorical:
            logging.info('Categorically-labelled graphs, using CATEGORICAL propagation scheme.')
    
    # Embed the nodes
    if categorical:
        es = WeisfeilerLehman()
        node_representations = es.fit_transform(X, num_iterations=num_iterations)
    else:
        es = ContinuousWeisfeilerLehman()
        node_representations = es.fit_transform(X, node_features=node_features, num_iterations=num_iterations)

    # Compute the Wasserstein distance
    print("Computing Wasserstein distance between conditions...")
    pairwise_distances = _compute_wasserstein_distance_geomloss(node_representations, categorical=categorical)
    return pairwise_distances

def wwl(X, node_features=None, num_iterations=3, sinkhorn=False, gamma=None):
    """
    Pairwise computation of the Wasserstein Weisfeiler-Lehman kernel for graphs in X.
    """
    D_W =  pairwise_wasserstein_distance(X, node_features = node_features, 
                                num_iterations=num_iterations, sinkhorn=sinkhorn)
    wwl = laplacian_kernel(D_W, gamma=gamma)
    return wwl
