import scipy as sp

from app.lib.twitter_search import TwitterSearch
from app.lib.word_tokenizer import WordTokenizer
from app.lib.graph_web import GraphWeb
from app.lib.embedding_analyzer import EmbeddingAnalyzer
from app.lib.embedding_analysis_plotter import EmbeddingAnalysisPlotter
from app.lib.network_builder import NetworkBuilder


if __name__ == "__main__":
    ################ Assignmen 1: Perform relevant Twitter searches
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

    ################ Assignment 2: Build document embeddings and classify into groups.
    # # 1) Build document embeddings
    # # Tokenize
    # word_tokenizer = WordTokenizer()

    # # Get the vectors
    # top_words_analyst_judgement, \
    #     train_vec_analyst_judgement, \
    #     test_vec_analyst_judgement, \
    #     train_target_analytics_judgement, \
    #     test_target_analyst_judgement = word_tokenizer.analyst_judgement('tweets', 30)
    
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

    # # 2) Classify, into manually assigned groups (manually encoded for this purpose).
    # tfidf_pred, tfidf_result = word_tokenizer.random_forest_classifier(train_vec_tfidf, test_vec_tfidf, train_target_tfidf, test_target_tfidf)
    # nn_pred, nn_result = word_tokenizer.random_forest_classifier(train_vec_nn, test_vec_nn, train_target_nn, test_target_nn)

    # print('TF-IDF score: ' + str(tfidf_result))
    # print('NN embedding score: ' + str(nn_result))

    ################ Assignment 3: Document grouping and building semantic web
    # # 1) Do MDS, KMeans, and visualize results
    # tfidf_docs_mds_array, tfidf_docs_kmeans_groups = EmbeddingAnalyzer(train_vec_tfidf).analyzer('tfidf', 'docs')
    # tfidf_tokens_mds_array, tfidf_tokens_kmeans_groups = EmbeddingAnalyzer(train_vec_tfidf.transpose()).analyzer('tfidf', 'tokens')
    # nn_docs_mds_array, nn_docs_kmeans_groups = EmbeddingAnalyzer(train_vec_nn).analyzer('nn', 'docs')
    # nn_full_docs_mds_array, nn_full_docs_kmeans_groups = EmbeddingAnalyzer(train_vec_nn).analyzer('nn_full', 'docs', mds=False)
    
    # EmbeddingAnalysisPlotter().plot_user_results(tfidf_docs_mds_array, tfidf_docs_kmeans_groups, 'tfidf_docs', user_df_tfidf)
    # EmbeddingAnalysisPlotter().plot_token_results(tfidf_tokens_mds_array, tfidf_tokens_kmeans_groups, 'tfidf_tokens')
    # EmbeddingAnalysisPlotter().plot_user_results(nn_docs_mds_array, nn_docs_kmeans_groups, 'nn_docs', user_df_nn)
    # EmbeddingAnalysisPlotter().plot_user_results(nn_full_docs_mds_array, nn_full_docs_kmeans_groups, 'nn_docs_full', user_df_nn)

    # # 2) Build graph from kmeans groups as web
    # GraphWeb(tfidf_docs_kmeans_groups[3], user_df_tfidf['username'].tolist()).build_graph('tfidf_10_groups')
    # GraphWeb(nn_docs_kmeans_groups[3], user_df_nn['username'].tolist()).build_graph('nn_10_groups')

    ################ Assignment 4
    NetworkBuilder().build_network()