import argparse

from app.lib.twitter_search import TwitterSearch
from app.lib.nn_tokenizer import NnTokenizer
from app.lib.tf_idf_tokenizer import TfIdfTokenizer
from app.lib.graph_web import GraphWeb
from app.lib.embedding_analyzer import EmbeddingAnalyzer
from app.lib.embedding_analysis_plotter import EmbeddingAnalysisPlotter
from app.lib.user_analyzer import UserAnalyzer
from app.lib.user_embedding_analyzer import UserEmbeddingAnalyzer


# Globals
COUNT_WORDS = 30
TWEETS_FILE = 'tweets'
SAMPLE_RATIO = 0.1


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--search', action="store_true", default=False, dest='search_flag')

    args = parser.parse_args()

    # ############### Assignment 4
    # 1) Perform new Twitter Search if running for the first time.
    if args.search_flag:
        print('[Runner] Performing new Twitter search...')
        TwitterSearch().search(
            query_list=[
                'violence',
                'guns',
                'mass shooting',
                'progun',
                'gun control',
                'second amendment',
                'debate',
                'politics',
                'political',
                'trump',
                'bernie',
                'biden',
                'warren',
                'pence',
                'election'
            ],
            count=1000,
            filename=TWEETS_FILE
        )

    # 2) User Analyzer: KNN based on retweets and replies; also builds users file
    rt_knn_model = UserAnalyzer().analyzer(read_file=True, filename_users='users')

    # 3) Build document embeddings and tokenize for more analysis
    user_df_tfidf, \
        tf_idf_vectorizer, \
        train_vec_tfidf, \
        test_vec_tfidf, \
        train_target_tfidf, \
        test_target_tfidf = TfIdfTokenizer().tf_idf_train(TWEETS_FILE, COUNT_WORDS)

    user_df_nn, \
        model_nn, \
        train_vec_nn, \
        test_vec_nn, \
        train_target_nn, \
        test_target_nn = NnTokenizer().nn_embeddings_train(TWEETS_FILE)

    # 3) Do KMeans as exploratory function
    tfidf_docs_mds_array, tfidf_docs_kmeans_groups = EmbeddingAnalyzer(train_vec_tfidf).analyzer('tfidf', 'docs')
    nn_docs_mds_array, nn_docs_kmeans_groups = EmbeddingAnalyzer(train_vec_nn).analyzer('nn', 'docs')

    EmbeddingAnalysisPlotter().plot_user_results(tfidf_docs_mds_array, tfidf_docs_kmeans_groups, 'tfidf_docs', user_df_tfidf)
    EmbeddingAnalysisPlotter().plot_user_results(nn_docs_mds_array, nn_docs_kmeans_groups, 'nn_docs', user_df_nn)

    # 4) Compare to previous assignment's random forest
    tfidf_rf_model = UserEmbeddingAnalyzer().rf_analyzer('tfidf', train_vec_tfidf, test_vec_tfidf, train_target_tfidf, test_target_tfidf)
    nn_rf_model = UserEmbeddingAnalyzer().rf_analyzer('nn', train_vec_nn, test_vec_nn, train_target_nn, test_target_nn)

    # 5) User embedding analyzer: KNN on embeddings
    tfidf_knn_model = UserEmbeddingAnalyzer().knn_analyzer('tfidf', train_vec_tfidf, test_vec_tfidf, train_target_tfidf, test_target_tfidf)
    nn_knn_model = UserEmbeddingAnalyzer().knn_analyzer('nn', train_vec_nn, test_vec_nn, train_target_nn, test_target_nn)

    # 6) Take most accurate model (TF-IDF KNN), classify sample including unknown users
    user_df_tfidf_full, vec_tfidf_full = TfIdfTokenizer().tf_idf_apply(TWEETS_FILE, tf_idf_vectorizer, sample_ratio=SAMPLE_RATIO)
    tfidf_full_pred = tfidf_knn_model.predict(vec_tfidf_full)

    # 7) Build network from MDS Euclidean distance
    tfidf_docs_mds_array_full = EmbeddingAnalyzer(vec_tfidf_full).analyzer('tfidf', 'all_docs', kmeans=False)
    print(user_df_tfidf_full.shape)
    print(user_df_tfidf_full.columns)

    # GraphWeb(tfidf_docs_kmeans_groups[3], user_df_tfidf['username'].tolist()).build_graph('tfidf_10_groups')
