import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import os
import re
import string

# Note: this requires nltk.download() first as described in the README.
# from nltk.book import *
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from collections import Counter, OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
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

    def _parse_words(self, text, stemming=False): 
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

        # perform word stemming if requested
        if stemming:
            ps = PorterStemmer()
            tokens = [ps.stem(word) for word in tokens]

        # recreate the document string from parsed words
        text = ''
        for token in tokens:
            text = text + ' ' + token

        return tokens, text

    def _get_train_test_data(self, filename):
        # Get df, and list of all users' tweets.
        tweets_by_user_df = self._user_grouper(filename)
        
        # Get user classes
        db_cols = ['class', 'user_description', 'username']
        user_class_df = jsonl_to_df('users', db_cols)
        user_class_df = user_class_df[['username','class']]

        tagged_df = pd.merge(tweets_by_user_df, user_class_df, left_on='username', right_on='username')

        train, test = train_test_split(tagged_df, test_size=0.2)
        train_target = train['class']
        test_target = test['class']
        return train, test, train_target, test_target

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

    def tf_idf(self, filename, count_words):
        print('[WordTokenizer] Getting TF-IDF vectors...')
        # Get df, and list of all users' tweets.
        tweets_by_user_df, tweets_by_user_df_test, train_target, test_target = self._get_train_test_data(filename)

        # NOTE: cleanup with stopwords makes it not english because the string removal isn't right.
        # all_stopwords = list(stopwords.words('english'))
        # all_stopwords.extend(['rt'])
        # for word in all_stopwords:
        #     tweets_by_user_df['tweets'] = tweets_by_user_df['tweets'].str.replace(word,'')

        tweets_by_user_list = tweets_by_user_df['tweets'].tolist()
        tweets_by_user_list_test = tweets_by_user_df_test['tweets'].tolist()

        # Calculate TF-IDF
        vectorizer = TfidfVectorizer()
        tweets_by_user_vec = vectorizer.fit_transform(tweets_by_user_list)
        tweets_by_user_vec_test = vectorizer.transform(tweets_by_user_list_test)

        # Change up matrices to get most important words
        feature_array = np.array(vectorizer.get_feature_names())
        tweets_by_user_sorted_vec = np.argsort(tweets_by_user_vec.toarray()).flatten()[::-1]

        # top_words_tfidf = feature_array[tweets_by_user_sorted_vec][:count_words]
        return tweets_by_user_df, tweets_by_user_vec, tweets_by_user_vec_test, train_target, test_target

    def nn_embeddings(self, filename, retrain=True):
        print('[WordTokenizer] Getting NN embedding vectors...')
        # NOTE: getting embeddings for entire corpus. Not splitting into train/test yet.
        #       The train/test split will occur outside of this function as the model develops next assignment.

        # Use DF for all tweets by user
        tweets_by_user_df, tweets_by_user_df_test, train_target, test_target = self._get_train_test_data(filename)

        # Format the documents
        tweets_by_user_list = tweets_by_user_df['tweets'].tolist()
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tweets_by_user_list)]
        tokens = []
        for doc in tweets_by_user_list:
            text_string = self._parse_doc(doc)
            doc_tokens, text_string = self._parse_words(text_string)
            tokens.append(doc_tokens)

        # Format test documents
        tweets_by_user_list_test = tweets_by_user_df_test['tweets'].tolist()
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tweets_by_user_list_test)]
        tokens_test = []
        for doc in tweets_by_user_list_test:
            text_string = self._parse_doc(doc)
            doc_tokens, text_string = self._parse_words(text_string)
            tokens_test.append(doc_tokens)

        model_filename = get_tmpfile('doc2vec_model')

        # Load the documents and train the model, if flag set to retrain
        if retrain:
            model = Doc2Vec(documents, vector_size=50, window=4, min_count=2, workers=4, epochs=40)
            model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
            model.save(model_filename)

        # Load the model
        model = Doc2Vec.load(model_filename)

        # vectorize training and test sets
        doc2vec_model_vectors_train = np.zeros((len(tokens), 50))
        for i in range(0, len(tokens)):
            doc2vec_model_vectors_train[i, ] = model.infer_vector(tokens[i]).transpose()

        doc2vec_model_vectors_test = np.zeros((len(tokens_test), 50))
        for i in range(0, len(tokens_test)):
            doc2vec_model_vectors_test[i, ] = model.infer_vector(tokens_test[i]).transpose()

        return tweets_by_user_df, model, doc2vec_model_vectors_train, doc2vec_model_vectors_test, train_target, test_target

    def random_forest_classifier(self, train_vec, test_vec, train_target, test_target):
        print('[WordTokenizer] Building classifier...')
        classifier = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = 5)
        classifier.fit(train_vec, train_target)
        classifier_pred = classifier.predict(test_vec)  # evaluate on test set
        classifier_results = round(metrics.f1_score(test_target, classifier_pred, average='macro'), 3)
        return classifier_pred, classifier_results
