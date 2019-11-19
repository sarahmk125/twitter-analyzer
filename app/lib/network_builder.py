import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import string

from app.lib.utils.jsonl import jsonl_to_df
from statistics import mean
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class NetworkBuilder(object):
    def __init__(self):
        pass

    def _numeric_classes(self, row):
        if row['class'] == 'U':
            return 0
        elif row['class'] == 'P':
            return 1
        elif row['class'] == 'N':
            return 3
        elif row['class'] == 'Y':
            return 4

    def _user_grouper(self, filename):
        # For each unique user, join all tweets into one tweet row in the new df.
        # Also, have count of number of tweets and total number of retweets
        db_cols = ['search_query', 'id_str', 'full_text', 'created_at', 'favorite_count', 'username', 'user_description']
        tweets_df = jsonl_to_df(filename, db_cols)
        users = list(tweets_df['username'].unique())
        tweets_by_user_df = pd.DataFrame(columns=['username', 'user_description', 'tweets'])

        # Iterate through all users.
        for i, user in enumerate(users):
            trunc_df = tweets_df[tweets_df['username'] == user]

            # Feature calculation: general vars
            num_tweets = trunc_df.shape[0]
            tweets_list = trunc_df["full_text"].tolist()
            total_letters_list = [len(tweet) for tweet in tweets_list]
            avg_total_letters = mean(total_letters_list)

            # Retweet ratio
            retweets_df = trunc_df[trunc_df['full_text'].str.contains('RT ')]
            num_retweets = retweets_df.shape[0]
            ratio_retweets = float(num_retweets) / float(num_tweets)

            # Replies ratio
            replies_df = trunc_df[trunc_df['full_text'].str.startswith('@')]
            num_replies = replies_df.shape[0]
            ratio_replies = float(num_replies) / float(num_tweets)

            # Capital letter ratio average across tweets
            capital_letters_list = [sum(1 for c in tweet if c.isupper()) for tweet in tweets_list]
            avg_capital_letters = mean(capital_letters_list)
            avg_ratio_capital_letters = mean([val / total_letters_list[i] for i, val in enumerate(capital_letters_list)])

            # Punctuation ratio average across tweets
            punctuation_list = [sum(1 for p in tweet if p in string.punctuation) for tweet in tweets_list]
            avg_punctuation_chars = mean(punctuation_list)
            avg_ratio_punctuation_chars = mean([val / total_letters_list[i] for i, val in enumerate(punctuation_list)])

            user_description = trunc_df['user_description'].tolist()[0]
            full_string = ' '.join(trunc_df["full_text"])
            tweets_by_user_df = tweets_by_user_df.append(
                {
                    'username': user,
                    'user_description': user_description,
                    'num_tweets': num_tweets,
                    'num_retweets': num_retweets,
                    'ratio_retweets': ratio_retweets,
                    'num_replies': num_replies,
                    'ratio_replies': ratio_replies,
                    'avg_total_letters': avg_total_letters,
                    'avg_capital_letters': avg_capital_letters,
                    'avg_ratio_capital_letters': avg_ratio_capital_letters,
                    'avg_punctuation_chars': avg_punctuation_chars,
                    'avg_ratio_punctuation_chars': avg_ratio_punctuation_chars,
                    'tweets': full_string
                },
                ignore_index=True)

        # Class DF as numeric classes
        db_cols_class = ['class', 'username']
        user_class_df = jsonl_to_df('users', db_cols_class)
        user_class_df['class'] = user_class_df.apply(lambda row: self._numeric_classes(row), axis=1)

        # Join dfs
        full_df = pd.merge(tweets_by_user_df, user_class_df, left_on='username', right_on='username')

        # Return the data frame with one row per user, tweets concatenated into one string.
        return full_df

    def _plot_retweet_behavior(self, df, column, title, filename):
        # Data aggregation
        df_bar = df.groupby([
                'class'
            ]).agg({
                column: 'mean',
            }).reset_index()

        x = list(df_bar['class'])
        y = list(df_bar[column])

        # Data order
        df_bar = df_bar.sort_values(by='class')

        # Plotting
        fig = plt.figure(figsize=(10, 5))
        fig.add_subplot(111)

        sns.barplot(x, y, color="lightcoral")
        plt.title(title, fontsize=16)
        plt.ylabel('Average Ratio', fontsize=10)
        plt.xlabel('Class', fontsize=10)
        plt.tight_layout()
        plt.savefig('./app/scripts/visuals/' + filename + '.png')

    def _knn(self, df):
        # Format data
        df_x = df[['ratio_retweets', 'ratio_replies', 'avg_ratio_capital_letters', 'avg_ratio_punctuation_chars']]
        x = df_x.values.tolist()
        y = df['class']

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=60)

        # Fit classifier
        knn_model = KNeighborsClassifier(n_neighbors=1)
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)
        print(classification_report(y_test, y_pred))

    def build_network(self, filename='tweets'):
        users_df = self._user_grouper(filename)
        self._plot_retweet_behavior(users_df, 'ratio_retweets', 'Ratio Retweets', 'avg_ratio_retweets_by_class')
        self._plot_retweet_behavior(users_df, 'ratio_replies', 'Ratio Replies', 'avg_ratio_replies_by_class')
        self._knn(users_df)
