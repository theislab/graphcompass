"""
Wasserstein Weisfeiler-Lehman (WWL) kernel implementation.

This module provides tools for computing graph similarities using the Wasserstein 
Weisfeiler-Lehman kernel, supporting both categorical and continuous graph embeddings.

Adapted from: https://github.com/BorgwardtLab/WWL/blob/master/src/wwl/wwl.py
"""

import sys
import logging

import numpy as np
import torch
from geomloss import SamplesLoss
from sklearn.preprocessing import OneHotEncoder
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

def _compute_wasserstein_distance_geomloss(label_sequences, categorical=False, blur=0.05, p=2):
    """Compute pairwise Wasserstein distances between graph node embeddings.

    Calculates the optimal transport distance between node embeddings using 
    the Sinkhorn algorithm. Automatically uses GPU acceleration if available.

    Args:
        label_sequences (list): List of node embeddings for each graph
        categorical (bool): Whether the node labels are categorical (discrete) or continuous embeddings.
        blur (float, optional): Sinkhorn smoothing parameter. Defaults to 0.05.
        p (int, optional): Power of the cost function. Defaults to 2 (squared Euclidean).

    Returns:
        numpy.ndarray: Symmetric matrix of pairwise Wasserstein distances

    Notes:
        This function differs from the legacy implementation in that it leverages
        the GeomLoss library for GPU-accelerated Sinkhorn computations, enabling 
        efficient optimal transport calculations even on large graphs. The legacy
        function uses the POT library and runs on CPU only, which can be slower 
        for large datasets. Additionally, this function handles both categorical
        and continuous node features in a unified manner via one-hot encoding 
        for discrete labels.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sinkhorn = SamplesLoss("sinkhorn", p=p, blur=blur)

    n = len(label_sequences)
    M = torch.zeros((n, n), device=device)

    if categorical:
        # Flatten all labels for one-hot encoder fitting
        all_labels = np.concatenate(label_sequences).reshape(-1, 1)
        enc = OneHotEncoder(sparse_output=False, dtype=np.float32)
        enc.fit(all_labels)

        # Encode all graphs now for speed
        encoded_sequences = [torch.tensor(enc.transform(seq.reshape(-1,1)), device=device) for seq in label_sequences]

    else:
        # Assume label_sequences are arrays of shape (n_nodes, features)
        encoded_sequences = [torch.tensor(seq, dtype=torch.float32, device=device) for seq in label_sequences]

    for i in range(n):
        a = torch.ones(encoded_sequences[i].shape[0], device=device) / encoded_sequences[i].shape[0]
        for j in range(i, n):
            b = torch.ones(encoded_sequences[j].shape[0], device=device) / encoded_sequences[j].shape[0]

            dist = sinkhorn(a, encoded_sequences[i], b, encoded_sequences[j])
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
            if 'label' not in g.vs.attribute_names() or not all(isinstance(label, (int, float)) for label in g.vs['label']):
                logging.info('Invalid categorical labels found: Switching to continuous propagation scheme using node degrees.')
                categorical = False
                break
        if categorical:
            logging.info('Valid categorical graph labels detected: Using categorical propagation scheme.')
    
    # Embed the nodes
    if categorical:
        es = WeisfeilerLehman()
        node_representations = es.fit_transform(X, num_iterations=num_iterations)
    else:
        es = ContinuousWeisfeilerLehman()
        node_representations = es.fit_transform(X, node_features=node_features, num_iterations=num_iterations)

    # Compute the Wasserstein distance
    logging.info("Computing pairwise Wasserstein distances between graph embeddings...")
    pairwise_distances = _compute_wasserstein_distance_geomloss(node_representations, categorical=categorical)
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
