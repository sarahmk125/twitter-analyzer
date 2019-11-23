import scipy as sp

from app.lib.twitter_search import TwitterSearch
from app.lib.word_tokenizer import WordTokenizer
from app.lib.graph_web import GraphWeb
from app.lib.embedding_analyzer import EmbeddingAnalyzer
from app.lib.embedding_analysis_plotter import EmbeddingAnalysisPlotter
from app.lib.network_builder import NetworkBuilder


if __name__ == "__main__":
    # ############### Assignment 4
    # TwitterSearch().search(
    #     query_list=[
    #         'violence',
    #         'guns',
    #         'mass shooting',
    #         'progun',
    #         'gun control',
    #         'second amendment',
    #         'debate',
    #         'politics',
    #         'political',
    #         'trump',
    #         'bernie',
    #         'biden',
    #         'warren',
    #         'pence',
    #         'election'
    #     ],
    #     count=1000,
    #     filename='tweets'
    # )

    # Network builder builds users file
    NetworkBuilder().build_network(read_file=True, filename_users='users')

    # # Documents
    # # 1) Build document embeddings and tokenize
    # word_tokenizer = WordTokenizer()
    # user_df_tfidf, \
    #     train_vec_tfidf, \
    #     test_vec_tfidf, \
    #     train_target_tfidf, \
    #     test_target_tfidf = word_tokenizer.tf_idf('tweets', 30)

    # user_df_nn, \
    #     model_nn, \
    #     train_vec_nn, \
    #     test_vec_nn, \
    #     train_target_nn, \
    #     test_target_nn = word_tokenizer.nn_embeddings(filename='tweets', retrain=True)

    # # 2) Do KMeans
    # tfidf_docs_mds_array, tfidf_docs_kmeans_groups = EmbeddingAnalyzer(train_vec_tfidf).analyzer('tfidf', 'docs')
    # tfidf_tokens_mds_array, tfidf_tokens_kmeans_groups = EmbeddingAnalyzer(train_vec_tfidf.transpose()).analyzer('tfidf', 'tokens')
    # nn_docs_mds_array, nn_docs_kmeans_groups = EmbeddingAnalyzer(train_vec_nn).analyzer('nn', 'docs')

    # EmbeddingAnalysisPlotter().plot_user_results(tfidf_docs_mds_array, tfidf_docs_kmeans_groups, 'tfidf_docs', user_df_tfidf)
    # EmbeddingAnalysisPlotter().plot_token_results(tfidf_tokens_mds_array, tfidf_tokens_kmeans_groups, 'tfidf_tokens')
    # EmbeddingAnalysisPlotter().plot_user_results(nn_docs_mds_array, nn_docs_kmeans_groups, 'nn_docs', user_df_nn)
