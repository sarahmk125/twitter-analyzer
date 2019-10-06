import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Note: this requires nltk.download() first as described in the README.
# from nltk.book import *
from nltk.corpus import stopwords
from nltk import word_tokenize, FreqDist
from collections import Counter


"""
Sources:
Loading JSONL: https://medium.com/@galea/how-to-love-jsonl-using-json-line-format-in-your-workflow-b6884f65175b
NLTK Reference: http://www.nltk.org/book/ch01.html
NLTK word counter reference: https://www.strehle.de/tim/weblog/archives/2015/09/03/1569
"""


def _load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data


def _jsonl_to_df(filename):
    jsonl_file = '../tweets/' + filename + '.jsonl'
    tweets_data = _load_jsonl(jsonl_file)
    db_data = []
    # Note: ignoring hashtags for now since they're nested
    db_cols = ['search_query', 'id_str', 'full_text', 'created_at', 'favorite_count', 'username', 'user_description']
    for d in tweets_data:
        db_data.append([])
        for col in db_cols:
            db_data[-1].append(d.get(col, float('nan')))

    tweets_df = pd.DataFrame(db_data, columns=db_cols)
    return tweets_df


def _word_tokenizer(tweets_df):
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


def explorer(filename):
    # Load the tweets
    tweets_df = _jsonl_to_df(filename)
    words_list = _word_tokenizer(tweets_df)

    # Plot
    words_counts = pd.Series(words_list).value_counts() #.plot(kind='bar')
    # print(v)
    # #plt.savefig('./visuals/most_common_words.pdf')
    words_counts = words_counts[:30, ]
    plt.figure(figsize=(10, 5))
    words_plot = sns.barplot(words_counts.index, words_counts.values, alpha=0.8)
    words_plot.set_xticklabels(words_plot.get_xticklabels(), rotation=45)
    plt.title('Most Common Words in Tweets')
    plt.ylabel('Occurrences', fontsize=12)
    plt.xlabel('Word', fontsize=12)
    plt.tight_layout()
    plt.savefig('./visuals/most_common_words.pdf')


if __name__ == "__main__":
    explorer('tweets')
