import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Note: this requires nltk.download() first as described in the README.
# from nltk.book import *
from nltk.corpus import stopwords
from nltk import word_tokenize, FreqDist
from app.lib.utils.jsonl import jsonl_to_df


class TweetPlotter(object):
    def __init__(self):
        pass

    def _word_counter(self, tweets_df):
        full_texts = tweets_df['full_text'].tolist()
        id_strings = tweets_df['id_str'].tolist()
        full_texts_dict = {}
        full_text_list = []

        all_stopwords = set(stopwords.words('english'))

        for i, text in enumerate(full_texts):
            # Tokenize
            words = word_tokenize(text)
            # Remove single chars, numbers, lowercase all, then remove stop words
            words = [word for word in words if len(word) > 1]
            words = [word for word in words if not word.isnumeric()]
            words = [word.lower() for word in words]
            words = [word for word in words if word not in all_stopwords]

            # Word distribution
            word_dist = FreqDist(words)
            # Only get actual words for now, not frequency. Might use later.
            word_list = [w[0] for w in word_dist.most_common(50)]

            # I don't use this now, but I probably will later
            ids = id_strings[i]
            full_texts_dict.update({ids: word_list})

            full_text_list.extend(words)

        return full_text_list

    def plot(self, filename='tweets'):
        print('[TweetPlotter] Visualizing most common words...')

        # Load the tweets
        tweets_df = jsonl_to_df(filename)
        words_list = self._word_counter(tweets_df)

        # Plot
        words_counts = pd.Series(words_list).value_counts()
        words_counts = words_counts[:30, ]
        plt.figure(figsize=(10, 5))
        words_plot = sns.barplot(words_counts.index, words_counts.values, alpha=0.8)
        words_plot.set_xticklabels(words_plot.get_xticklabels(), rotation=45)
        plt.title('Most Common Words in Tweets')
        plt.ylabel('Occurrences', fontsize=12)
        plt.xlabel('Word', fontsize=12)
        plt.tight_layout()
        plt.savefig('app/scripts/visuals/most_common_words.png')
