import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy


# Note: this requires nltk.download() first as described in the README.
# from nltk.book import *
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
#from nltk import word_tokenize, FreqDist
from nltk.tokenize import TreebankWordTokenizer
from collections import Counter, OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer


"""
Sources:
Loading JSONL: https://medium.com/@galea/how-to-love-jsonl-using-json-line-format-in-your-workflow-b6884f65175b
NLTK Reference: http://www.nltk.org/book/ch01.html
NLTK word counter reference: https://www.strehle.de/tim/weblog/archives/2015/09/03/1569
"""


class WordTokenizer(object):
    def __init__(self):
        pass

    def _load_jsonl(self, input_path) -> list:
        """
        Read list of objects from a JSON lines file.
        """
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.rstrip('\n|\r')))
        print('[WordTokenizer] Loaded {} records from {}'.format(len(data), input_path))
        return data

    def _jsonl_to_df(self, filename):
        jsonl_file = 'app/tweets/' + filename + '.jsonl'
        tweets_data = self._load_jsonl(jsonl_file)
        db_data = []
        # Note: ignoring hashtags for now since they're nested
        db_cols = ['search_query', 'id_str', 'full_text', 'created_at', 'favorite_count', 'username', 'user_description']
        for d in tweets_data:
            db_data.append([])
            for col in db_cols:
                db_data[-1].append(d.get(col, float('nan')))

        tweets_df = pd.DataFrame(db_data, columns=db_cols)
        return tweets_df

    def _user_grouper(self, filename):
        # For each unique user, join all tweets into one tweet row in the new df.
        tweets_df = self._jsonl_to_df(filename)
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

    def analyst_judgement(self, filename, count_words):
        # Get df, and list of all users' tweets.
        tweets_by_user_df = self._user_grouper(filename)

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
            tweets_by_user_vec.append(vec_list)

        tweets_by_user_array = np.array(tweets_by_user_vec)
        return top_words, tweets_by_user_array

    def tf_idf(self, filename, count_words):
        # Get df, and list of all users' tweets.
        tweets_by_user_df = self._user_grouper(filename)

        # NOTE: cleanup with stopwords makes it not english because the string removal isn't right.
        # all_stopwords = list(stopwords.words('english'))
        # all_stopwords.extend(['rt'])
        # for word in all_stopwords:
        #     tweets_by_user_df['tweets'] = tweets_by_user_df['tweets'].str.replace(word,'')

        tweets_by_user_list = tweets_by_user_df['tweets'].tolist()

        # Calculate TF-IDF
        vectorizer = TfidfVectorizer()
        tweets_by_user_vec = vectorizer.fit_transform(tweets_by_user_list)

        # Change up matrices to get most important words
        feature_array = np.array(vectorizer.get_feature_names())
        tweets_by_user_sorted_vec = np.argsort(tweets_by_user_vec.toarray()).flatten()[::-1]

        top_words = feature_array[tweets_by_user_sorted_vec][:count_words]
        return top_words, tweets_by_user_vec




        # # Tokenize whole corpus
                # all_tweets = ' '.join(tweets_by_user_df['tweets'])
        # tokenizer = TreebankWordTokenizer()
        # lexicon = sorted(tokenizer.tokenize(all_tweets.lower()))
        # all_stopwords = set(stopwords.words('english'))
        # lexicon = [x for x in lexicon if x not in all_stopwords]
        # lexicon_counts = Counter(lexicon)


        # # Assign vector for each user
        # zero_vector = OrderedDict((token, 0) for token in lexicon)
        # doc_vectors = []
        # for row in tweets_by_user_df.iterrows():
        #     vec = copy.copt(zero_vector)
        #     tokens = tokenizer.tokenize(row['tweets'].lower())
        #     token_counts = Counter(tokens)
        #     for key, value in token_counts.items():
        #         vec[key] = value / len(lexicon)





        # TF-IDF Vector calculation
        # Count of each word in a document / # of docs in which word occurs

        


    # def _nltk_tokenizer(self, tweets_df):
    #     full_texts = tweets_df['full_text'].tolist()
    #     id_strings = tweets_df['id_str'].tolist()
    #     full_texts_dict = {}
    #     full_text_list = []

    #     all_stopwords = set(stopwords.words('english'))

    #     print(f'[WordTokenizer] Tokenizing full text of {len(full_texts)} tweets...')
    #     for i, text in enumerate(full_texts):
    #         # Tokenize
    #         words = word_tokenize(text)
    #         # Remove single chars, numbers, lowercase all, then remove stop words
    #         words = [word for word in words if len(word) > 1]
    #         words = [word for word in words if not word.isnumeric()]
    #         words = [word.lower() for word in words]
    #         words = [word for word in words if word not in all_stopwords]

    #         # Word distribution
    #         word_dist = FreqDist(words)
    #         # Only get actual words for now, not frequency. Might use later.
    #         word_list = [w[0] for w in word_dist.most_common(50)]

    #         # I don't use this now, but I probably will later
    #         ids = id_strings[i]
    #         full_texts_dict.update({ids: word_list})
    #         full_text_list.extend(words)

    #     return full_text_list, full_texts_dict, word_dist

    # def word_tokenizer(self, filename):
    #     # Load the tweets
    #     tweets_df = self._jsonl_to_df(filename)
    #     words_list, words_dict, word_dist = self._nltk_tokenizer(tweets_df)

    #     # Apply bag of words, maybe
    #     return words_list, words_dict, word_dist

