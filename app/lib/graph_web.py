import pandas as pd
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt

from itertools import combinations


class GraphWeb(object):
    def __init__(self, groups, labels):
        self.groups = groups
        self.labels = labels
        self.nodes = self._get_nodes()
        self.edges = self._get_edges()

    def _get_nodes(self):
        return self.labels

    def _get_edges(self):
        edges = []
        for group in self.groups:
            indices = [i for i, val in enumerate(self.groups) if val == group]
            labels_trunc = [val for i, val in enumerate(self.labels) if i in indices]
            new_edges = list(combinations(labels_trunc, 2))
            edges.extend(new_edges)

        return edges

    def build_graph(self, name):
        # Build
        print(f'[GraphWeb] Building graph: {name}...')
        g = nx.Graph()
        g.add_nodes_from(self.nodes)
        g.add_edges_from(self.edges)

        # Visualize
        print(f'[GraphWeb] Visualizing graph: {name}...')
        nx.draw(g)
        plt.savefig(f'app/scripts/visuals/graph_web_{name}.png')

    # Euclidean distance
    # dist = np.linalg.norm(a-b)
