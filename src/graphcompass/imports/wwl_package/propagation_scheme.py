######## This file is copied from https://github.com/BorgwardtLab/WWL/blob/master/src/wwl/propagation_scheme.py ########

# -----------------------------------------------------------------------------
# This file contains the propagation schemes for categorically labeled and 
# continuously attributed graphs.
#
# November 2019, M. Togninalli
# -----------------------------------------------------------------------------
import numpy as np

from sklearn.preprocessing import scale
from sklearn.base import TransformerMixin

import argparse
import igraph as ig
import os

import copy
from collections import defaultdict
from typing import List
from tqdm import tqdm
from scipy.sparse import csr_matrix, diags


####################
# Weisfeiler-Lehman
####################
class WeisfeilerLehman(TransformerMixin):
    """
    Class that implements the Weisfeiler-Lehman transform
    Credits: Christian Bock and Bastian Rieck
    """
    def __init__(self):
        self._relabel_steps = defaultdict(dict)
        self._label_dict = {}
        self._last_new_label = -1
        self._preprocess_relabel_dict = {}
        self._results = defaultdict(dict)
        self._label_dicts = {}

    def _reset_label_generation(self):
        self._last_new_label = -1

    def _get_next_label(self):
        self._last_new_label += 1
        return self._last_new_label

    def _relabel_graphs(self, X: List[ig.Graph]):
        num_unique_labels = 0
        preprocessed_graphs = []
        for i, g in enumerate(X):
            x = g.copy()
            
            if not 'label' in x.vs.attribute_names():
                x.vs['label'] = list(map(str, [l for l in x.vs.degree()]))           
            labels = x.vs['label']
            

            new_labels = []
            for label in labels:
                if label in self._preprocess_relabel_dict.keys():
                    new_labels.append(self._preprocess_relabel_dict[label])
                else:
                    self._preprocess_relabel_dict[label] = self._get_next_label()
                    new_labels.append(self._preprocess_relabel_dict[label])
            x.vs['label'] = new_labels
            self._results[0][i] = (labels, new_labels)
            self._label_sequences[i][:, 0] = new_labels
            preprocessed_graphs.append(x)
        self._reset_label_generation()
        return preprocessed_graphs

    def fit_transform(self, X: List[ig.Graph], num_iterations: int=3, return_sequences=True):
        self._label_sequences = [
            np.full((len(g.vs), num_iterations + 1), np.nan) for g in X
        ]
        X = self._relabel_graphs(X)
        for it in np.arange(1, num_iterations+1, 1):
            self._reset_label_generation()
            self._label_dict = {}
            for i, g in enumerate(X):
                # Get labels of current interation
                current_labels = g.vs['label']

                # Get for each vertex the labels of its neighbors
                neighbor_labels = self._get_neighbor_labels(g, sort=True)

                # Prepend the vertex label to the list of labels of its neighbors
                merged_labels = [[b]+a for a,b in zip(neighbor_labels, current_labels)]

                # Generate a label dictionary based on the merged labels
                self._append_label_dict(merged_labels)

                # Relabel the graph
                new_labels = self._relabel_graph(g, merged_labels)
                self._relabel_steps[i][it] = { idx: {old_label: new_labels[idx]} for idx, old_label in enumerate(current_labels) }
                g.vs['label'] = new_labels

                self._results[it][i] = (merged_labels, new_labels)
                self._label_sequences[i][:, it] = new_labels
            self._label_dicts[it] = copy.deepcopy(self._label_dict)
        if return_sequences:
            return self._label_sequences
        else:
            return self._results

    def _relabel_graph(self, X: ig.Graph, merged_labels: list):
        new_labels = []
        for merged in merged_labels:
            new_labels.append(self._label_dict['-'.join(map(str,merged))])
        return new_labels

    def _append_label_dict(self, merged_labels: List[list]):
        for merged_label in merged_labels:
            dict_key = '-'.join(map(str,merged_label))
            if dict_key not in self._label_dict.keys():
                self._label_dict[ dict_key ] = self._get_next_label()

    def _get_neighbor_labels(self, X: ig.Graph, sort: bool=True):
            neighbor_indices = [[n_v.index for n_v in X.vs[X.neighbors(v.index)]] for v in X.vs]
            neighbor_labels = []
            for n_indices in neighbor_indices:
                if sort:
                    neighbor_labels.append( sorted(X.vs[n_indices]['label']) )
                else:
                    neighbor_labels.append( X.vs[n_indices]['label'] )
            return neighbor_labels

####################
# Continuous Weisfeiler-Lehman
####################

class ContinuousWeisfeilerLehman(TransformerMixin):
    """
    Class that implements the continuous Weisfeiler-Lehman propagation scheme
    """
    def __init__(self):
        self._results = defaultdict(dict)
        self._label_sequences = []

    def _preprocess_graphs(self, X: List[ig.Graph]):
        """
        Load graphs from gml files.
        """
        # initialize
        node_features = []
        adj_mat = []
        n_nodes = []

        # Iterate across graphs and load initial node features
        for graph in X:
            if not 'label' in graph.vs.attribute_names():
                graph.vs['label'] = list(map(str, [l for l in graph.vs.degree()]))    
            # Get features and adjacency matrix
            node_features_cur = np.asarray(graph.vs['label']).astype(float).reshape(-1, 1)
            adj_mat_cur = csr_matrix(graph.get_adjacency_sparse())
            # Load features
            node_features.append(node_features_cur)
            adj_mat.append(adj_mat_cur)
            n_nodes.append(adj_mat_cur.shape[0])

        # By default, keep degree or label as features, if other features shall
        # to be used (e.g. the one from the TU Dortmund website), 
        # provide them to the fit_transform function.

        n_nodes = np.asarray(n_nodes)
        node_features = np.asarray(node_features, dtype=object)

        return node_features, adj_mat, n_nodes

    def _create_adj_avg(self, adj_cur: csr_matrix) -> csr_matrix:
        '''
        Create adjacency matrix using sparse operations.
        '''
        deg = np.array(adj_cur.sum(axis=1)).flatten()

        # Adjust degrees where needed
        deg[deg!=1] -= 1

        deg = 1 / deg
        deg_mat = diags(deg)

        # Normalize adjacency matrix
        adj_cur = deg_mat @ adj_cur  # Sparse multiplication
        
        return adj_cur

    def fit_transform(self, X: List[ig.Graph], node_features = None, num_iterations: int=3):
        """
        Transform a list of graphs into their node representations. 
        Node features should be provided as a numpy array.
        """
        print("Embedding nodes; pre-processing...")
        node_features_labels, adj_mat, n_nodes = self._preprocess_graphs(X)
        if node_features is None:
            node_features = node_features_labels

        print("Embedding nodes; feature scaling...")
        node_features_data = scale(np.concatenate(node_features, axis=0), axis = 0)
        splits_idx = np.cumsum(n_nodes).astype(int)
        node_features_split = np.vsplit(node_features_data,splits_idx)		
        node_features = node_features_split[:-1]

        # Generate the label sequences for h iterations
        print("Embedding nodes; generating label sequences...")
        n_graphs = len(node_features)
        self._label_sequences = []
        for i in tqdm(range(n_graphs)):
            graph_feat = []

            for it in range(num_iterations+1):
                if it == 0:
                    graph_feat.append(node_features[i])
                else:
                    adj_cur = adj_mat[i] + csr_matrix(np.identity(adj_mat[i].shape[0]))
                    adj_cur = self._create_adj_avg(adj_cur)

                    adj_cur.setdiag(0)
                    graph_feat_cur = 0.5 * (adj_cur @ graph_feat[it-1] + graph_feat[it-1])
                    graph_feat.append(graph_feat_cur)

            self._label_sequences.append(np.concatenate(graph_feat, axis=1))
        return self._label_sequences