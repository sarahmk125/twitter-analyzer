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
        

            