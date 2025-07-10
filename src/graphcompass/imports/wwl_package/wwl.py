"""
Wasserstein Weisfeiler-Lehman (WWL) kernel implementation.

This module provides tools for computing graph similarities using the Wasserstein 
Weisfeiler-Lehman kernel, supporting both categorical and continuous graph embeddings.

Adapted from: https://github.com/BorgwardtLab/WWL/blob/master/src/wwl/wwl.py
"""

import sys
import logging

import torch
from geomloss import SamplesLoss
from sklearn.metrics.pairwise import laplacian_kernel

from .propagation_scheme import WeisfeilerLehman, ContinuousWeisfeilerLehman

logging.basicConfig(level=logging.INFO)

def logging_config(level='DEBUG'):
    """Set the logging level for the application.

    Configures the global logging level to control the verbosity of log messages.

    Args:
        level (str, optional): Logging level. Defaults to 'DEBUG'.
            Typical values include 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    """
    logging.basicConfig(level=logging.getLevelName(level.upper()))

def _compute_wasserstein_distance_geomloss(label_sequences, blur=0.05, p=2):
    """Compute pairwise Wasserstein distances between graph node embeddings.

    Calculates the optimal transport distance between node embeddings using 
    the Sinkhorn algorithm. Automatically uses GPU acceleration if available.

    Args:
        label_sequences (list): List of node embeddings for each graph
        blur (float, optional): Sinkhorn smoothing parameter. Defaults to 0.05.
        p (int, optional): Power of the cost function. Defaults to 2 (squared Euclidean).

    Returns:
        numpy.ndarray: Symmetric matrix of pairwise Wasserstein distances
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

def pairwise_wasserstein_distance(X, node_features=None, num_iterations=3, enforce_continuous=False):
    """Compute pairwise Wasserstein distances between graph embeddings.

    Determines the appropriate embedding scheme (categorical or continuous) 
    and computes the Wasserstein distances between graph representations.

    Args:
        X (list): List of graphs to compare
        node_features (array-like, optional): Pre-computed node features for continuous graphs
        num_iterations (int, optional): Number of iterations for graph embedding. Defaults to 3.
        enforce_continuous (bool, optional): Force use of continuous embedding scheme. Defaults to False.

    Returns:
        numpy.ndarray: Matrix of pairwise Wasserstein distances between graphs
    """
    # First check if the graphs are continuous vs categorical
    categorical = True
    if enforce_continuous:
        logging.info('Continuous embedding enforced: Using continuous propagation scheme.')
        categorical = False
    elif node_features is not None:
        logging.info('Continuous node features detected: Using continuous propagation scheme.')
        categorical = False
    else:
        for g in X:
            if 'label' not in g.vs.attribute_names():
                logging.info('No categorical labels found: Switching to continuous propagation scheme using node degrees.')
                categorical = False
                break
        if categorical:
            logging.info('Categorical graph labels detected: Using categorical propagation scheme.')
    
    # Embed the nodes
    if categorical:
        es = WeisfeilerLehman()
        node_representations = es.fit_transform(X, num_iterations=num_iterations)
    else:
        es = ContinuousWeisfeilerLehman()
        node_representations = es.fit_transform(X, node_features=node_features, num_iterations=num_iterations)

    # Compute the Wasserstein distance
    logging.info("Computing pairwise Wasserstein distances between graph embeddings...")
    pairwise_distances = _compute_wasserstein_distance_geomloss(node_representations)
    return pairwise_distances

def wwl(X, node_features=None, num_iterations=3, gamma=None):
    """Compute the Wasserstein Weisfeiler-Lehman (WWL) kernel for a set of graphs.

    Combines Wasserstein distance computation with a Laplacian kernel to 
    measure graph similarities.

    Args:
        X (list): List of graphs to compare
        node_features (array-like, optional): Pre-computed node features for continuous graphs
        num_iterations (int, optional): Number of iterations for graph embedding. Defaults to 3.
        gamma (float, optional): Scaling parameter for the Laplacian kernel. Defaults to None.

    Returns:
        numpy.ndarray: Kernel matrix representing graph similarities
    """
    D_W =  pairwise_wasserstein_distance(X, node_features = node_features, 
                                num_iterations=num_iterations)
    wwl = laplacian_kernel(D_W, gamma=gamma)
    return wwl
