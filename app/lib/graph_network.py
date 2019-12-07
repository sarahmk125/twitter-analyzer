import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from sklearn.metrics.pairwise import euclidean_distances


class GraphNetwork(object):
    def __init__(self, preds, vec_full, user_df):
        self.preds = preds
        self.vec_full = vec_full
        self.labels = user_df['username'].tolist()
        self.user_df = user_df
        self.nodes = self._get_nodes()
        self.edges = self._get_edges()

    def _get_nodes(self):
        return self.labels

    def _get_edges(self):
        edges = []

        # Form edges by euclidean distances
        euc_distances = euclidean_distances(self.vec_full, self.vec_full)
        # Don't count vector exactly the same, ie. distance of zero
        euc_distances = np.where(euc_distances == 0, 9999, euc_distances)
        # Take closest 3 nodes
        euc_distances_sort = euc_distances.argsort(axis=1)
        euc_distances_sort = euc_distances_sort[:, 0:3]

        for i in range(0, euc_distances_sort.shape[0]):
            for j in range(0, euc_distances_sort.shape[1]):
                new_edge = (i, euc_distances_sort[i, j])
                edges.append(new_edge)

        return edges

    def build_graph(self, name):
        # Build
        print(f'[GraphNetwork] Building graph: {name}...')
        g = nx.Graph()
        g.add_nodes_from(self.nodes)
        g.add_edges_from(self.edges)

        # Visualize
        print(f'[GraphNetwork] Visualizing graph: {name}...')
        fig, ax = plt.subplots()
        pos = nx.spring_layout(g)  # positions for all nodes

        # Nodes and edges
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=[i for i, x in enumerate(self.user_df['class'].tolist()) if (x == 'U' and self.preds[i] == 'M')],
            node_color='#F0AC1C',
            node_size=150,
            alpha=0.9)
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=[i for i, x in enumerate(self.user_df['class'].tolist()) if (x == 'U' and self.preds[i] == 'R')],
            node_color='#15D6DF',
            node_size=150,
            alpha=0.9)
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=[i for i, x in enumerate(self.user_df['class'].tolist()) if x == 'M'],
            node_color='r',
            node_size=150,
            alpha=0.9)
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=[i for i, x in enumerate(self.user_df['class'].tolist()) if x == 'R'],
            node_color='b',
            node_size=150,
            alpha=0.9)

        nx.draw_networkx_edges(g, pos, edgelist=self.edges, width=1.5, alpha=0.8)

        # Plot neatness
        plt.title('KNN Predictions Network')
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Unknown - MAGA', markerfacecolor='#F0AC1C', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Unknown - Resistance', markerfacecolor='#15D6DF', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='MAGA', markerfacecolor='r', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Resistance', markerfacecolor='blue', markersize=10)
        ]
        ax.legend(handles=legend_elements, loc='upper left')

        plt.savefig(f'app/scripts/visuals/graph_network_{name}.png')
