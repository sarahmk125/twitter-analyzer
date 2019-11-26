import pandas as pd
import numpy as np
import copy
import re
import string

# Note: this requires nltk.download() first as described in the README.
# from nltk.book import *
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from collections import Counter, OrderedDict
from sklearn.model_selection import train_test_split
from app.lib.utils.jsonl import jsonl_to_df


"""
Sources:
Loading JSONL: https://medium.com/@galea/how-to-love-jsonl-using-json-line-format-in-your-workflow-b6884f65175b
NLTK Reference: http://www.nltk.org/book/ch01.html
NLTK word counter reference: https://www.strehle.de/tim/weblog/archives/2015/09/03/1569
"""


class WordTokenizer(object):
    def __init__(self):
        pass

    def _user_grouper(self, filename):
        # For each unique user, join all tweets into one tweet row in the new df.
        db_cols = ['search_query', 'id_str', 'full_text', 'created_at', 'favorite_count', 'username', 'user_description']
        tweets_df = jsonl_to_df(filename, db_cols)
        users = list(tweets_df['username'].unique())
        tweets_by_user_df = pd.DataFrame(columns=['username', 'user_description', 'tweets'])

        # Iterate through all users.
        for i, user in enumerate(users):
            trunc_df = tweets_df[tweets_df['username'] == user]
            user_description = trunc_df['user_description'].tolist()[0]
            string = ' '.join(trunc_df["full_text"])
            tweets_by_user_df = tweets_by_user_df.append({'username': user, 'user_description': user_description, 'tweets': string}, ignore_index=True)

        # Return the data frame with one row per user, tweets concatenated into one string.
        return tweets_by_user_df

    def _parse_doc(self, text):
        text = text.lower()
        text = re.sub(r'&(.)+', "", text)  # no & references
        text = re.sub(r'pct', 'percent', text)  # replace pct abreviation
        text = re.sub(r"[^\w\d'\s]+", '', text)  # no punct except single quote
        text = re.sub(r'[^\x00-\x7f]', r'', text)  # no non-ASCII strings

        # Omit words that are all digits
        if text.isdigit():
            text = ""

        # # Get rid of escape codes
        # for code in codelist:
        #     text = re.sub(code, ' ', text)

        # Replace multiple spacess with one space
        text = re.sub('\s+', ' ', text)

        return text

    def _parse_words(self, text): 
        # split document into individual words
        tokens = text.split()
        re_punc = re.compile('[%s]' % re.escape(string.punctuation))

        # remove punctuation from each word
        tokens = [re_punc.sub('', w) for w in tokens]

        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]

        # filter out tokens that are one or two characters long
        tokens = [word for word in tokens if len(word) > 2]

        # filter out tokens that are more than twenty characters long
        tokens = [word for word in tokens if len(word) < 21]

        # recreate the document string from parsed words
        text = ''
        for token in tokens:
            text = text + ' ' + token

        return tokens, text

    def _get_train_test_data(self, filename, only_known=True):
        # Get df, and list of all users' tweets.
        tweets_by_user_df = self._user_grouper(filename)

        # Get user classes
        db_cols = ['class', 'user_description', 'username']
        user_class_df = jsonl_to_df('users', db_cols)
        user_class_df = user_class_df[['username', 'class']]

        tagged_df = pd.merge(tweets_by_user_df, user_class_df, left_on='username', right_on='username')

        if only_known:
            tagged_df = tagged_df[tagged_df['class'] != 'U']

        train, test = train_test_split(tagged_df, test_size=0.2, random_state=60)
        train_target = train['class']
        test_target = test['class']
        return train, test, train_target, test_target

    def _get_all_classes(self, filename, sample_ratio=1):
        # Get df, and list of all users' tweets.
        tweets_by_user_df = self._user_grouper(filename)

        # Get user classes
        db_cols = ['class', 'user_description', 'username']
        user_class_df = jsonl_to_df('users', db_cols)
        user_class_df = user_class_df[['username', 'class']]

        tagged_df = pd.merge(tweets_by_user_df, user_class_df, left_on='username', right_on='username')
        tagged_df = tagged_df.sample(frac=sample_ratio, replace=False, random_state=60)
        return tagged_df

    def analyst_judgement(self, filename, count_words):
        print('[WordTokenizer] Getting analyst judgement vectors...')
        # Get df, and list of all users' tweets.
        tweets_by_user_df, tweets_by_user_df_test, train_target, test_target = self._get_train_test_data(filename)

        # Tokenize whole corpus, where lexicon counts are counts of each word across all tweets in the file
        tokenizer = TreebankWordTokenizer()
        all_stopwords = list(stopwords.words('english'))
        # Manual exclusion of twitter specific stuff and punctuation
        all_stopwords.extend(['rt', '#', '\'', '@', '!', '``', '\'\'', '\'s', '?', '`', ':', ',', 'https'])

        all_tweets = ' '.join(tweets_by_user_df['tweets'])
        lexicon = sorted(tokenizer.tokenize(all_tweets.lower()))
        lexicon = [x for x in lexicon if x not in all_stopwords]

        # Get top X words
        lexicon_counts = Counter(lexicon)
        top_words = [w[0] for w in lexicon_counts.most_common(count_words)]

        # Vectorization: 
        # Take X most common words, that is the size of the vector for each document.
        # The value in each vector: # times that word is present in the doc / total doc length
        # Apply this for the training and test dfs
        zero_vector = OrderedDict((word, 0) for word in top_words)
        tweets_by_user_vec = []
        for index, row in tweets_by_user_df.iterrows():
            vec = copy.copy(zero_vector)
            tokens = tokenizer.tokenize(row['tweets'].lower())
            token_counts = Counter(tokens)

            # Iterate through all words in document, seeing if they should be assigned to value in vector
            for key, value in token_counts.items():
                try:
                    vec[key] = value / len(tokens)
                except KeyError:
                    # Word is not top word, not doing anything with that information.
                    continue

            # Transform ordered dict to list
            vec_list = [i[1] for i in vec.items()]
            vec_array = np.array(vec_list)
            tweets_by_user_vec.append(vec_array)

        tweets_by_user_vec_test = []
        for index, row in tweets_by_user_df_test.iterrows():
            vec = copy.copy(zero_vector)
            tokens = tokenizer.tokenize(row['tweets'].lower())
            token_counts = Counter(tokens)

            # Iterate through all words in document, seeing if they should be assigned to value in vector
            for key, value in token_counts.items():
                try:
                    vec[key] = value / len(tokens)
                except KeyError:
                    # Word is not top word, not doing anything with that information.
                    continue

            # Transform ordered dict to list
            vec_list = [i[1] for i in vec.items()]
            vec_array = np.array(vec_list)
            tweets_by_user_vec_test.append(vec_array)

        tweets_by_user_array = np.array(tweets_by_user_vec)
        tweets_by_user_array_test = np.array(tweets_by_user_vec_test)
        return top_words, tweets_by_user_array, tweets_by_user_array_test, train_target, test_target
