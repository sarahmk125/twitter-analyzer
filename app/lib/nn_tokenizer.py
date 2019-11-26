import numpy as np

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from app.lib.word_tokenizer import WordTokenizer


class NnTokenizer(WordTokenizer):
    def __init__(self):
        super().__init__()

    def _format_docs(self, tweets_by_user_df):
        tweets_by_user_list = tweets_by_user_df['tweets'].tolist()
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tweets_by_user_list)]
        tokens = []
        for doc in tweets_by_user_list:
            text_string = self._parse_doc(doc)
            doc_tokens, text_string = self._parse_words(text_string)
            tokens.append(doc_tokens)
        return tokens, documents

    def _vectorize_docs(self, tokens, model):
        doc2vec_model_vectors = np.zeros((len(tokens), 50))
        for i in range(0, len(tokens)):
            doc2vec_model_vectors[i, ] = model.infer_vector(tokens[i]).transpose()
        return doc2vec_model_vectors

    def _calculate_nn(self, tweets_by_user_df, model=None):
        tokens, documents = self._format_docs(tweets_by_user_df)

        if not model:
            model = Doc2Vec(documents, vector_size=50, window=4, min_count=2, workers=4, epochs=40)
            model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
            doc2vec_model_vectors = self._vectorize_docs(tokens, model)
            return model, doc2vec_model_vectors

        doc2vec_model_vectors = self._vectorize_docs(tokens, model)
        return model, doc2vec_model_vectors

    def nn_embeddings_train(self, filename):
        print('[NnTokenizer] Getting NN embedding vectors for new model...')
        tweets_by_user_df, tweets_by_user_df_test, train_target, test_target = self._get_train_test_data(filename)

        # Format the documents
        model, doc2vec_model_vectors_train = self._calculate_nn(tweets_by_user_df)
        model, doc2vec_model_vectors_test = self._calculate_nn(tweets_by_user_df_test, model=model)

        return tweets_by_user_df, model, doc2vec_model_vectors_train, doc2vec_model_vectors_test, train_target, test_target

    def nn_embeddings_apply(self, filename, model, sample_ratio=1):
        print('[NnTokenizer] Getting NN embedding vectors for existing model...')
        tweets_by_user_df = self._get_all_classes(filename, sample_ratio)

        # Calculate embeddings
        model, doc2vec_model_vectors = self._calculate_nn(tweets_by_user_df, model=model)
        return tweets_by_user_df, doc2vec_model_vectors
