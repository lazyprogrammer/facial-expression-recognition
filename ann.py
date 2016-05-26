import numpy as np
import matplotlib.pyplot as plt

from util import getData, softmax, cost, y2indicator, error_rate, relu
from sklearn.utils import shuffle


class ANN(object):
    def __init__(self, M):
        self.M = M

    def fit(self, X, Y, learning_rate=10e-8, reg=10e-12, epochs=10000, show_fig=False):
        X, Y = shuffle(X, Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        Tvalid = y2indicator(Yvalid)
        X, Y = X[:-1000], Y[:-1000]

        N, D = X.shape
        K = len(set(Y))
        T = y2indicator(Y)
        self.W1 = np.random.randn(D, self.M) / np.sqrt(D + self.M)
        self.b1 = np.zeros(self.M)
        self.W2 = np.random.randn(self.M, K) / np.sqrt(self.M + K)
        self.b2 = np.zeros(K)

        costs = []
        best_validation_error = 1
        for i in xrange(epochs):
                # forward propagation and cost calculation
                pY, Z = self.forward(X)

                # gradient descent step
                self.W2 -= learning_rate*(Z.T.dot(pY - T) + reg*self.W2)
                self.b2 -= learning_rate*((pY - T).sum(axis=0) + reg*self.b2)
                self.W1 -= learning_rate*(X.T.dot((pY - T).dot(self.W2.T) * (Z > 0)) + reg*self.W1)
                self.b1 -= learning_rate*(np.sum((pY - T).dot(self.W2.T) * (Z > 0), axis=0) + reg*self.b1)

                if i % 10 == 0:
                    pYvalid = self.forward(Xvalid)
                    c = cost(Tvalid, pYvalid)
                    costs.append(c)
                    e = error_rate(Yvalid, np.argmax(pYvalid, axis=1))
                    print "i:", i, "cost:", c, "error:", e
                    if e < best_validation_error:
                        best_validation_error = e
        print "best_validation_error:", best_validation_error

        if show_fig:
            plt.plot(costs)
            plt.show()


    def forward(self, X):
    	Z = relu(X.dot(self.W1) + self.b1)
        return softmax(Z.dot(self.W1) + self.b2), Z

    def predict(self, X):
        pY, _ = self.forward(X)
        return np.argmax(pY, axis=1)

    def score(self, X, Y):
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)


def main():
    X, Y = getData()
    
    model = ANN(3000)
    model.fit(X, Y, show_fig=True)
    model.score(X, Y)
    # scores = cross_val_score(model, X, Y, cv=5)
    # print "score mean:", np.mean(scores), "stdev:", np.std(scores)

if __name__ == '__main__':
    main()
