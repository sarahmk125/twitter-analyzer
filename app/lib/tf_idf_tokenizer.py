from sklearn.feature_extraction.text import TfidfVectorizer
from app.lib.word_tokenizer import WordTokenizer


class TfIdfTokenizer(WordTokenizer):
    def __init__(self):
        super().__init__()

    def _calculate_tf_idf(self, tweets_by_user_df, vectorizer=None):
        tweets_by_user_list = tweets_by_user_df['tweets'].tolist()

        if not vectorizer:
            vectorizer = TfidfVectorizer()
            tweets_by_user_vec = vectorizer.fit_transform(tweets_by_user_list)
            return vectorizer, tweets_by_user_vec

        tweets_by_user_vec = vectorizer.transform(tweets_by_user_list)
        return vectorizer, tweets_by_user_vec

    def tf_idf_train(self, filename, count_words):
        print('[TfIdfTokenizer] Getting TF-IDF vectors for new model...')
        # Get df, and list of all users' tweets.
        tweets_by_user_df, tweets_by_user_df_test, train_target, test_target = self._get_train_test_data(filename)

        # Calculate TF-IDF
        vectorizer, tweets_by_user_vec = self._calculate_tf_idf(tweets_by_user_df)
        vectorizer, tweets_by_user_vec_test = self._calculate_tf_idf(tweets_by_user_df_test, vectorizer=vectorizer)

        # Change up matrices to get most important words
        # feature_array = np.array(vectorizer.get_feature_names())
        # tweets_by_user_sorted_vec = np.argsort(tweets_by_user_vec.toarray()).flatten()[::-1]
        # top_words_tfidf = feature_array[tweets_by_user_sorted_vec][:count_words]

        return tweets_by_user_df, vectorizer, tweets_by_user_vec, tweets_by_user_vec_test, train_target, test_target

    def tf_idf_apply(self, filename, vectorizer, sample_ratio=1):
        print('[TfIdfTokenizer] Getting TF-IDF vectors for existing vectorizer...')
        tweets_by_user_df = self._get_all_classes(filename, sample_ratio)

        # Calculate TF-IDF
        vectorizer, tweets_by_user_vec = self._calculate_tf_idf(tweets_by_user_df, vectorizer=vectorizer)
        return tweets_by_user_df, tweets_by_user_vec
