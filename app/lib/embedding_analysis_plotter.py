import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt



class EmbeddingAnalysisPlotter(object):
    def __init__(self):
        pass

    def _plot_mds(self, mds_array, name, labels=None, label_flag=False):
        # Plotting first 50 documents with labels, as more than that clutters the plot.
        # Description: Each dot is a user. Cannot discern meaning.
        if label_flag:
            plt.figure()
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
        plt.title(f'MDS Results: {name.upper()} Embeddings', fontsize=14)
        plt.savefig(f'app/scripts/visuals/mds_{name}_nolabels.png')

        plt.close()

    def _get_kmeans_user_data(self, mapped_users):
        usernames = [i[0] for i in mapped_users]
        user_groups = [i[1] for i in mapped_users]

        groups = list(set(user_groups))
        group_counts = []
        for group in groups:
            group_counts.append(sum(user_group == group for user_group in user_groups))
        
        return usernames, user_groups, groups, group_counts

    def _get_kmeans_plot_data(self, y_kmeans):
        groups = list(set(y_kmeans.tolist()))
        group_counts = []
        for group in groups:
            group_counts.append(sum(result == group for result in y_kmeans.tolist()))
        
        return groups, group_counts

    def _plot_kmeans_scatter(self, name, array, y_kmeans):
        groups, group_counts = self._get_kmeans_plot_data(y_kmeans)  

        fig, ax = plt.subplots()
        x = array[:,0]
        y = array[:,1]
        scatter = ax.scatter(x, y, c=y_kmeans.tolist())
        legend = ax.legend(*scatter.legend_elements(),
                            loc="lower left", title="Groups")
        ax.add_artist(legend)
        plt.title(f'KMeans Scatter Results: {name.upper()} Embeddings', fontsize=14)
        plt.savefig(f'app/scripts/visuals/kmeans_scatter_{len(groups)}_{name}.png')

        plt.close()

    def _plot_kmeans_results(self, name, y_kmeans, ylabel, title):
        groups, group_counts = self._get_kmeans_plot_data(y_kmeans)    
    
        # Plot number of users in each kmeans group
        plt.figure()

        plt.bar(groups, group_counts)
        plt.xlabel('KMeans Group', fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        plt.title(title, fontsize=14)

        plt.savefig(f'app/scripts/visuals/kmeans_{len(groups)}_{name}.png')
        plt.close()

    def _plot_kmeans_class_comparison(self, name, mapped_users, user_df):
        usernames, user_groups, groups, group_counts = self._get_kmeans_user_data(mapped_users)

        # Plot kmeans groups by manually coded class
        fig, axs = plt.subplots(2,2, figsize=(10,8))
        fig.suptitle(f'Subplots of KMeans Group Distribution Within Manually Assigned Classes: {name.upper()}')

        user_class = user_df['class'].tolist()
        user_class.sort()
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
        plt.close()

    def plot_user_results(self, mds_array, kmeans_groups, name, user_df):
        # Get list of users in this set
        user_list = user_df['username'].tolist()

        print(f'[EmbeddingAnalysisPlotter] Visualizing MDS on users for {name}...')
        self._plot_mds(mds_array, name, user_list, True)

        print(f'[EmbeddingAnalysisPlotter] Visualizing KMeans on users for {name}...')
        for kmeans_grouping in kmeans_groups:
            mapped_users = [(user_list[i], group) for i, group in enumerate(kmeans_grouping.tolist())]
            self._plot_kmeans_results(name, kmeans_grouping, 'Count Users', f'Users per KMeans Group: {name.upper()} Embeddings')
            self._plot_kmeans_class_comparison(name, mapped_users, user_df)
            self._plot_kmeans_scatter(name, mds_array, kmeans_grouping)

    def plot_token_results(self, mds_array, kmeans_groups, name):
        print(f'[EmbeddingAnalysisPlotter] Visualizing MDS on tokens for {name}...')
        self._plot_mds(mds_array, name)

        print(f'[EmbeddingAnalysisPlotter] Visualizing KMeans on tokens for {name}...')
        for kmeans_grouping in kmeans_groups:
            self._plot_kmeans_results(name, kmeans_grouping, 'Count Tokens', f'Tokens per KMeans Group: {name.upper()} Embeddings')
            self._plot_kmeans_scatter(name, mds_array, kmeans_grouping)
