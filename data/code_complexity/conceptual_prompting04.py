import numpy as np



class LogisticRegression:

    def __init__(self, learning_rate=0.01, num_iterations=1000):

        self.learning_rate = learning_rate

        self.num_iterations = num_iterations



    def _sigmoid(self, z):

        return 1 / (1 + np.exp(-z))

    

    def _loss(self, y_true, y_pred):

        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))



    def fit(self, X, y):

        self.weights = np.zeros(X.shape[1])

        self.bias = 0

        

        for _ in range(self.num_iterations):

            linear_model = np.dot(X, self.weights) + self.bias

            y_pred = self._sigmoid(linear_model)

            dw = (1 / X.shape[0]) * np.dot(X.T, (y_pred - y))

            db = (1 / X.shape[0]) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw

            self.bias -= self.learning_rate * db



    def predict_proba(self, X):

        linear_model = np.dot(X, self.weights) + self.bias

        return self._sigmoid(linear_model)



    def predict(self, X, threshold=0.5):

        return (self.predict_proba(X) >= threshold).astype(int)
