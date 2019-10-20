from app.lib.twitter_search import TwitterSearch
from app.lib.word_tokenizer import WordTokenizer


if __name__ == "__main__":
    # Perform relevant twitter searches; already complete from assignment 1
    # TwitterSearch().search(
    #     query_list=[
    #         'violence',
    #         'guns',
    #         'mass shooting',
    #         'progun',
    #         'gun control',
    #         'second amendment'
    #     ],
    #     count=1000
    # )

    # Tokenize
    # # NOTE: vectorization_analyst_judgement is a 2D np array (list of lists type thing)
    # # NOTE: vectorization_tf_df is a scipy csr_matrix.

    word_tokenizer = WordTokenizer()

    # Get the vectors
    top_words_analyst_judgement, train_vec_analyst_judgement, test_vec_analyst_judgement, train_target_analytics_judgement, test_target_analyst_judgement = word_tokenizer.analyst_judgement('tweets', 30)
    train_vec_tfidf, test_vec_tfidf, train_target_tfidf, test_target_tfidf = word_tokenizer.tf_idf('tweets', 30)
    model_nn, train_vec_nn, test_vec_nn, train_target_nn, test_target_nn = word_tokenizer.nn_embeddings(filename='tweets', retrain=False)

    # Classify
    tfidf_pred, tfidf_result = word_tokenizer.random_forest_classifier(train_vec_tfidf, test_vec_tfidf, train_target_tfidf, test_target_tfidf)
    nn_pred, nn_result = word_tokenizer.random_forest_classifier(train_vec_nn, test_vec_nn, train_target_nn, test_target_nn)

    print('TF-IDF score: ' + str(tfidf_result))
    print('NN embedding score: ' + str(nn_result))

