from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


class UserEmbeddingAnalyzer(object):
    def __init__(self):
        pass

    def knn_analyzer(self, name, X_train, X_test, y_train, y_test, test_size=0.2):
        print(f'[UserEmbeddingAnalyzer] Running KNN with embeddings {name}...')

        knn_model = KNeighborsClassifier(n_neighbors=1)
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)

        print(classification_report(y_test, y_pred))
        return knn_model

    def rf_analyzer(self, name, train_vec, test_vec, train_target, test_target):
        print(f'[UserEmbeddingAnalyzer] Running RF with embeddings {name}...')

        classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=60)
        classifier.fit(train_vec, train_target)
        classifier_pred = classifier.predict(test_vec)  # evaluate on test set

        print(classification_report(test_target, classifier_pred))
        return classifier
