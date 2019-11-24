import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string

from app.lib.utils.jsonl import jsonl_to_df, df_to_jsonl
from statistics import mean
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class UserAnalyzer(object):
    def __init__(self):
        pass

    def _user_grouper(self, filename):
        # For each unique user, join all tweets into one tweet row in the new df.
        # Also, have count of number of tweets and total number of retweets
        print('[UserAnalyzer] Grouping users and assigning features...')

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

            # Get class for user
            if '#maga' in user_description.lower() or '#maga' in full_string.lower() or 'RT @BernieSanders' in full_string or '#elizabethwarren' in full_string.lower():
                classif = 'M'
            elif '#theresistance' in user_description.lower() or '#maga' in full_string.lower() or 'RT @realDonaldTrump' in full_string:
                classif = 'R'
            else:
                classif = 'U'

            tweets_by_user_df = tweets_by_user_df.append(
                {
                    'username': user,
                    'user_description': user_description,
                    'class': classif,
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

        # Return the data frame with one row per user, tweets concatenated into one string.
        df_to_jsonl(tweets_by_user_df, 'users')
        return tweets_by_user_df

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

    def _knn(self, df, test_size=0.2):
        print('[UserAnalyzer] Running KNN...')
        # Format data
        df_x = df[['ratio_retweets', 'ratio_replies', 'avg_ratio_capital_letters', 'avg_ratio_punctuation_chars']]
        x = df_x.values.tolist()
        y = df['class']

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=60)

        # Fit classifier
        knn_model = KNeighborsClassifier(n_neighbors=1)
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)
        print(classification_report(y_test, y_pred))
        return knn_model

    def analyzer(self, filename_tweets='tweets', filename_users='users', read_file=False):
        print('[UserAnalyzer] Starting to build and classify...')
        if read_file:
            users_df = jsonl_to_df(filename_users)
        else:
            users_df = self._user_grouper(filename_tweets)

        # Filter out unknown
        print(f"Number users unknown: {len(users_df[users_df['class'] == 'U']['class'].tolist())}")
        print(f"Number users MAGA: {len(users_df[users_df['class'] == 'M']['class'].tolist())}")
        print(f"Number users Resistance: {len(users_df[users_df['class'] == 'R']['class'].tolist())}")

        # Only run KNN on known tags
        users_df_known = users_df[users_df['class'] != 'U']
        self._plot_retweet_behavior(users_df_known, 'ratio_retweets', 'Ratio Retweets', 'avg_ratio_retweets_by_class')
        self._plot_retweet_behavior(users_df_known, 'ratio_replies', 'Ratio Replies', 'avg_ratio_replies_by_class')
        knn_model = self._knn(users_df_known, test_size=0.2)
        return knn_model
