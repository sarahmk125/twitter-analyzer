import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.manifold import MDS
from scipy.spatial.distance import squareform,pdist
from app.lib.utils.jsonl import jsonl_to_df
from sklearn.cluster import KMeans


class Ontology(object):
    def __init__(self, matrix):
        # Transform sparse matrix
        if type(matrix) == np.ndarray:
            self.matrix = matrix 
        else:
            self.matrix = np.squeeze(np.asarray(matrix.todense())).astype(np.float64)

    def _mds(self):
        embedding = MDS(n_components=2, metric=False)
        mds_array = embedding.fit_transform(self.matrix)
        return mds_array

    def _plot_mds(self, mds_array, name, labels):
        # Plotting first 50 documents with labels, as more than that clutters the plot.
        # Description: Each dot is a user. Cannot discern meaning.
        x = mds_array[:,0][0:49]
        y = mds_array[:,1][0:49]
        plt.scatter(x, y)
        for i, label in enumerate(labels[0:49]):
           plt.annotate(label, (x[i], y[i]))

        plt.title(f'MDS Results for {name.upper()} Embeddings, First 50', fontsize=14)
        plt.savefig(f'app/scripts/visuals/mds_{name}.png')

        # Plot without labels
        plt.figure()
        x = mds_array[:,0]
        y = mds_array[:,1]
        plt.scatter(x, y)
        plt.title(f'MDS Results for {name.upper()} Embeddings', fontsize=14)
        plt.savefig(f'app/scripts/visuals/mds_{name}_nolabels.png')

    def _kmeans(self, clusters, user_list):
        kmeans = KMeans(n_clusters=clusters, random_state=0).fit(self.matrix)
        y_kmeans = kmeans.predict(self.matrix)
        mapped_users = [(user_list[i], group) for i, group in enumerate(y_kmeans.tolist())]
        return mapped_users
    
    def _plot_kmeans(self, name, mapped_users, user_df):
        usernames = [i[0] for i in mapped_users]
        user_groups = [i[1] for i in mapped_users]

        groups = list(set(user_groups))
        group_counts = []
        for group in groups:
            group_counts.append(sum(user_group == group for user_group in user_groups))
        
        # Plot number of users in each kmeans group
        plt.figure()

        plt.bar(groups, group_counts)
        plt.xlabel('KMeans Group', fontsize=10)
        plt.ylabel('Count Users', fontsize=10)
        plt.title('Users per KMeans Group', fontsize=14)

        plt.savefig(f'app/scripts/visuals/kmeans_{len(groups)}_{name}.png')

        # Plot kmeans groups by manually coded class
        fig, axs = plt.subplots(2,2, figsize=(10,8))
        fig.suptitle('Subplots of KMeans Group Distribution Within Manually Assigned Classes')

        user_class = user_df['class'].tolist()
        for i, cl in enumerate(list(set(user_class))):
            # Assign full class; this is kind of ugly, could be a funtion
            if cl == 'Y':
                full_class = 'Trump Supporter'
            elif cl == 'N':
                full_class = 'Trump Opposer'
            elif cl == 'U':
                full_class = 'Unknown, Not Political'
            elif cl == 'P':
                full_class = 'Political, Unknown Affiliation'
            else:
                full_class = '?'

            df_trunc = user_df[user_df['class'] == cl]
            usernames_trunc = df_trunc['username'].tolist()
            mapped_users_new = [item for item in mapped_users if item[0] in usernames_trunc]

            usernames_new = [i[0] for i in mapped_users_new]
            user_groups_new = [i[1] for i in mapped_users_new]

            groups_new = list(set(user_groups_new))
            group_counts_new = []
            for group in groups_new:
                group_counts_new.append(sum(user_group == group for user_group in user_groups_new))
            
            # This is ugly but not sure how to iterate through plots in the correct way otherwise
            if i == 0:
                x = 0
                y = 0
            elif i == 1:
                x = 0
                y = 1
            elif i == 2:
                x = 1
                y = 0
            elif i == 3:
                x = 1
                y = 1
            else:
                x = None
                y = None 

            axs[x,y].bar(groups_new, group_counts_new)
            axs[x,y].set_title(f'Users per Group in Class {full_class}', fontsize=10)

        fig.savefig(f'app/scripts/visuals/kmeans_classes_{len(groups)}_{name}.png')

    def build(self, name, user_df):
        user_list = user_df['username'].tolist()
        #user_class = user_df['class'].tolist()

        print('[Ontology] Performing and plotting MDS...')
        mds_array = self._mds()
        self._plot_mds(mds_array, name, user_list)

        print('[Ontology] Performing and visualizing KMeans...')
        mapped_users_2_groups = self._kmeans(2, user_list)
        mapped_users_3_groups = self._kmeans(3, user_list)
        mapped_users_4_groups = self._kmeans(4, user_list)

        self._plot_kmeans(name, mapped_users_2_groups, user_df)
        self._plot_kmeans(name, mapped_users_3_groups, user_df)
        self._plot_kmeans(name, mapped_users_4_groups, user_df)





        