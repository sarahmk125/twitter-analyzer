from app.lib.twitter_search import TwitterSearch
from app.lib.word_tokenizer import WordTokenizer

if __name__ == "__main__":
    # Perform relevant twitter searches
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
    top_words_analyst_judgement, vectorization_analyst_judgement = WordTokenizer().analyst_judgement('tweets', 30)
    top_words_tf_idf, vectorization_tf_idf = WordTokenizer().tf_idf('tweets', 30)

    # NOTE: vectorization_analyst_judgement is a 2D np array (list of lists type thing)
    # NOTE: vectorization_tf_df is a scipy csr_matrix.
    print(top_words_analyst_judgement)
    print(vectorization_analyst_judgement[0])
