import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from app.lib.utils.jsonl import jsonl_to_df
from sklearn.cluster import KMeans


class EmbeddingAnalyzer(object):
    def __init__(self, matrix):
        # Transform to type array
        if type(matrix) == np.ndarray:
            self.matrix = matrix
        else:
            self.matrix = matrix.toarray()

    def _mds(self):
        embedding = TSNE(n_components=2)
        mds_array = embedding.fit_transform(self.matrix)
        return mds_array

    def _kmeans(self, clusters, array):
        kmeans = KMeans(n_clusters=clusters, random_state=0).fit(array)
        y_kmeans = kmeans.predict(array)
        return y_kmeans
    
    # def _get_kmeans_user_data(self, mapped_users):
    #     usernames = [i[0] for i in mapped_users]
    #     user_groups = [i[1] for i in mapped_users]

    #     groups = list(set(user_groups))
    #     group_counts = []
    #     for group in groups:
    #         group_counts.append(sum(user_group == group for user_group in user_groups))
        
    #     return usernames, user_groups, groups, group_counts

    # def _get_kmeans_plot_data(self, y_kmeans):
    #     groups = list(set(y_kmeans.tolist()))
    #     group_counts = []
    #     for group in groups:
    #         group_counts.append(sum(result == group for result in y_kmeans.tolist()))
        
    #     return groups, group_counts

    def analyzer(self, name, transform, mds=True):
        print(f'[Ontology] Performing MDS on {transform} for {name}...')
        mds_array = self._mds()

        if mds:
            print(f'[Ontology] Performing KMeans on {transform} for {name}...')
            kmeans_groups = [self._kmeans(2, mds_array), self._kmeans(4, mds_array), self._kmeans(8, mds_array), self._kmeans(10, mds_array)]
            return mds_array, kmeans_groups
        
        print(f'[Ontology] Performing KMeans on {transform} for {name}...')
        kmeans_groups = [self._kmeans(2, self.matrix), self._kmeans(4, self.matrix), self._kmeans(8, self.matrix), self._kmeans(10, self.matrix)]
        return mds_array, kmeans_groups
        

            