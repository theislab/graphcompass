######## This file is adapted from https://github.com/BorgwardtLab/WWL/blob/master/src/wwl/wwl.py ########

import sys
import logging

import torch
from geomloss import SamplesLoss
from sklearn.metrics.pairwise import laplacian_kernel

from .propagation_scheme import WeisfeilerLehman, ContinuousWeisfeilerLehman

logging.basicConfig(level=logging.INFO)

def logging_config(level='DEBUG'):
    """Configure logging level.

    Args:
        level: Logging level (default: 'DEBUG')
    """
    logging.basicConfig(level=logging.getLevelName(level.upper()))

def _compute_wasserstein_distance_geomloss(label_sequences, blur=0.05, p=2):
    """Compute pairwise Wasserstein distances between graph node embeddings.

    Uses GeomLoss, automatically selecting GPU if available.

    Args:
        label_sequences: Node embeddings for each graph
        blur: Sinkhorn smoothing parameter
        p: Cost function power (default: Euclidean squared)

    Returns:
        Pairwise distance matrix
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

    Args:
        X: List of graphs
        node_features: Node features for continuous graphs
        num_iterations: Propagation scheme iterations
        enforce_continuous: Force continuous embedding scheme
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
    pairwise_distances = _compute_wasserstein_distance_geomloss(node_representations)
    return pairwise_distances

def wwl(X, node_features=None, num_iterations=3, gamma=None):
    """Compute Wasserstein Weisfeiler-Lehman kernel for graph set.

    Args:
        X: List of graphs
        node_features: Optional node features
        num_iterations: Propagation scheme iterations
        gamma: Laplacian kernel parameter
    """
    D_W =  pairwise_wasserstein_distance(X, node_features = node_features, 
                                num_iterations=num_iterations)
    wwl = laplacian_kernel(D_W, gamma=gamma)
    return wwl
