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
    db_cols = ['class', 'username', 'user_description']
    for d in tweets_data:
        db_data.append([])
        for col in db_cols:
            db_data[-1].append(d.get(col, float('nan')))

    tweets_df = pd.DataFrame(db_data, columns=db_cols)
    return tweets_df


def explorer(filename):
    # Load the tweets
    users_df = _jsonl_to_df(filename)
    
    def full_class_names(row):
        if row['class'] == 'U': return 'Unknown, not political'
        if row['class'] == 'P': return 'Unknown, political'
        if row['class'] == 'N': return 'Against Trump'
        if row['class'] == 'Y': return 'Trump supporter'
    
    users_df['class_full'] = users_df.apply(lambda row: full_class_names(row), axis=1)

    # Plot
    class_counts = pd.Series(users_df['class_full']).value_counts()
    plt.figure(figsize=(10, 5))
    users_plot = sns.barplot(class_counts.index, class_counts.values, alpha=0.8)
    users_plot.set_xticklabels(users_plot.get_xticklabels(), rotation=45)
    plt.title('Political Affiliation Distribtion')
    plt.ylabel('Number of Useres', fontsize=12)
    plt.xlabel('Tagged Affiliation', fontsize=12)
    plt.tight_layout()
    plt.savefig('./visuals/user_distribution.pdf')


if __name__ == "__main__":
    explorer('users')
