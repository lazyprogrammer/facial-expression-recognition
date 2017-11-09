from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from util import getData, getBinaryData, error_rate, relu, init_weight_and_bias
from sklearn.utils import shuffle



def rmsprop(cost, params, lr, mu, decay, eps):
    grads = T.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        # cache
        ones = np.ones_like(p.get_value(), dtype=np.float32)
        c = theano.shared(ones)
        new_c = decay*c + (np.float32(1.0) - decay)*g*g

        # momentum
        zeros = np.zeros_like(p.get_value(), dtype=np.float32)
        m = theano.shared(zeros)
        new_m = mu*m - lr*g / T.sqrt(new_c + eps)

        # param update
        new_p = p + new_m

        # append the updates
        updates.append((c, new_c))
        updates.append((m, new_m))
        updates.append((p, new_p))
    return updates


class HiddenLayer(object):
    def __init__(self, M1, M2, an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        W, b = init_weight_and_bias(M1, M2)
        self.W = theano.shared(W, 'W_%s' % self.id)
        self.b = theano.shared(b, 'b_%s' % self.id)
        self.params = [self.W, self.b]

    def forward(self, X):
        return relu(X.dot(self.W) + self.b)


class ANN(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, Y, learning_rate=1e-3, mu=0.9, decay=0.9, reg=0, eps=1e-10, epochs=100, batch_sz=30, show_fig=False):
        learning_rate = np.float32(learning_rate)
        mu = np.float32(mu)
        decay = np.float32(decay)
        reg = np.float32(reg)
        eps = np.float32(eps)

        # make a validation set
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = Y.astype(np.int32)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]

        # initialize hidden layers
        N, D = X.shape
        K = len(set(Y))
        self.hidden_layers = []
        M1 = D
        count = 0
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1
        W, b = init_weight_and_bias(M1, K)
        self.W = theano.shared(W, 'W_logreg')
        self.b = theano.shared(b, 'b_logreg')

        # collect params for later use
        self.params = [self.W, self.b]
        for h in self.hidden_layers:
            self.params += h.params

        # set up theano functions and variables
        thX = T.fmatrix('X')
        thY = T.ivector('Y')
        pY = self.th_forward(thX)

        rcost = reg*T.sum([(p*p).sum() for p in self.params])
        cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY])) + rcost
        prediction = self.th_predict(thX)

        # actual prediction function
        self.predict_op = theano.function(inputs=[thX], outputs=prediction)
        cost_predict_op = theano.function(inputs=[thX, thY], outputs=[cost, prediction])

        updates = rmsprop(cost, self.params, learning_rate, mu, decay, eps)
        train_op = theano.function(
            inputs=[thX, thY],
            updates=updates
        )

        n_batches = N // batch_sz
        costs = []
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

                train_op(Xbatch, Ybatch)

                if j % 20 == 0:
                    c, p = cost_predict_op(Xvalid, Yvalid)
                    costs.append(c)
                    e = error_rate(Yvalid, p)
                    print("i:", i, "j:", j, "nb:", n_batches, "cost:", c, "error rate:", e)
        
        if show_fig:
            plt.plot(costs)
            plt.show()

    def th_forward(self, X):
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return T.nnet.softmax(Z.dot(self.W) + self.b)

    def th_predict(self, X):
        pY = self.th_forward(X)
        return T.argmax(pY, axis=1)

    def predict(self, X):
        return self.predict_op(X)


def main():
    X, Y = getData()
    # X, Y = getBinaryData()
    model = ANN([2000, 1000])
    model.fit(X, Y, show_fig=True)

if __name__ == '__main__':
    main()
