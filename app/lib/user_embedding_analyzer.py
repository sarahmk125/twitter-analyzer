from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


class UserEmbeddingAnalyzer(object):
    def __init__(self):
        pass

    def _knn(self, name, X_train, X_test, y_train, y_test, test_size=0.2):
        print(f'[UserEmbeddingAnalyzer] Running KNN with embeddings {name}...')
        knn_model = KNeighborsClassifier(n_neighbors=1)
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)
        print(classification_report(y_test, y_pred))

    def analyzer(self, name, X_train, X_test, y_train, y_test):
        print(f'[EmbeddingAnalyzer] Classifying with {name}...')
        self._knn(name, X_train, X_test, y_train, y_test, test_size=0.2)
