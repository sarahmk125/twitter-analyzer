import argparse

from app.lib.twitter_search import TwitterSearch
from app.lib.word_tokenizer import WordTokenizer
# from app.lib.graph_web import GraphWeb
from app.lib.embedding_analyzer import EmbeddingAnalyzer
from app.lib.embedding_analysis_plotter import EmbeddingAnalysisPlotter
from app.lib.user_analyzer import UserAnalyzer
from app.lib.user_embedding_analyzer import UserEmbeddingAnalyzer


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
            filename='tweets'
        )

    # 2) User Analyzer: KNN based on retweets and replies; also builds users file
    rt_knn_model = UserAnalyzer().analyzer(read_file=True, filename_users='users')

    # 3) Build document embeddings and tokenize for more analysis
    word_tokenizer = WordTokenizer()
    user_df_tfidf, \
        train_vec_tfidf, \
        test_vec_tfidf, \
        train_target_tfidf, \
        test_target_tfidf = word_tokenizer.tf_idf('tweets', 30)

    user_df_nn, \
        model_nn, \
        train_vec_nn, \
        test_vec_nn, \
        train_target_nn, \
        test_target_nn = word_tokenizer.nn_embeddings(filename='tweets')

    # 3) Do KMeans as exploratory function
    tfidf_docs_mds_array, tfidf_docs_kmeans_groups = EmbeddingAnalyzer(train_vec_tfidf).analyzer('tfidf', 'docs')
    nn_docs_mds_array, nn_docs_kmeans_groups = EmbeddingAnalyzer(train_vec_nn).analyzer('nn', 'docs')

    EmbeddingAnalysisPlotter().plot_user_results(tfidf_docs_mds_array, tfidf_docs_kmeans_groups, 'tfidf_docs', user_df_tfidf)
    EmbeddingAnalysisPlotter().plot_user_results(nn_docs_mds_array, nn_docs_kmeans_groups, 'nn_docs', user_df_nn)

    # 4) User embedding analyzer: KNN on embeddings
    tfidf_knn_model = UserEmbeddingAnalyzer().analyzer('tfidf', train_vec_tfidf, test_vec_tfidf, train_target_tfidf, test_target_tfidf)
    nn_knn_model = UserEmbeddingAnalyzer().analyzer('nn', train_vec_nn, test_vec_nn, train_target_nn, test_target_nn)

    # 5) Take most accurate model (TF-IDF KNN), classify unknown users, build network from MDS Euclidean distance
    user_df_tfidf_full, \
        vec_tfidf_full = word_tokenizer.tf_idf('tweets', 30, get_all_data=True)
    tfidf_docs_mds_array_full = EmbeddingAnalyzer(vec_tfidf_full).analyzer('tfidf', 'all_docs', kmeans=False)

    # only take a portion for the above so it doesn't take forever?
    # build graph euclidean distance
    # compare to random forrest?
