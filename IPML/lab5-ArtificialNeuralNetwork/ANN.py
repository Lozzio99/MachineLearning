from typing import Callable, Any, Tuple, Union

import scipy.io
from numpy import ndarray
from sklearn.model_selection import train_test_split

from randInitializeWeights import randInitializeWeights
from predict import predict
from checkNNGradients import checkNNGradients
from nnCostFunction import nnCostFunction
from fmincg import fmincg
import numpy as np


class ANN:

    def __init__(self, n_in, n_hidden, n_out, random=True):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.random = random
        self.t1 = None
        self.t2 = None
        self.nn_params = None
        self.X = None
        self.Y = None
        self.cost = None
        self.reset()

    def _initialise_loading_(self):
        mat = scipy.io.loadmat('../data/debugweights.mat')
        self.t1 = mat['Theta1']
        Theta1_1d = np.reshape(self.t1, self.t1.size, order='F')
        self.t2 = mat['Theta2']
        Theta2_1d = np.reshape(self.t2, self.t2.size, order='F')
        self.nn_params = np.hstack((Theta1_1d, Theta2_1d))
        return self

    def _initialise_random_(self):
        self.t1 = randInitializeWeights(self.n_in, self.n_hidden)
        self.t2 = randInitializeWeights(self.n_hidden, self.n_out)
        self.t1 = np.reshape(self.t1, self.t1.size, order='F')
        self.t2 = np.reshape(self.t2, self.t2.size, order='F')
        self.nn_params = np.hstack((self.t1, self.t2))
        return self

    def set_data(self, X, Y):
        self.X = X
        self.Y = Y
        self.Y = np.expand_dims(self.Y, axis=1)
        return self

    def reset(self):
        self.cost = []
        if self.random:
            return self._initialise_random_()
        else:
            return self._initialise_loading_()

    def predict(self, X=None):
        return predict(self.t1, self.t2, (self.X if X is None else X))

    def prediction_accuracy(self, X=None, Y=None):
        if X is None and Y is None:
            return (self.predict(X) == Y).mean() * 100
        else:
            return (self.predict() == self.Y).mean() * 100

    def debug(self, debug=True):
        checkNNGradients(0, debug)
        checkNNGradients(3, debug)
        self._initialise_loading_()  # import loaded weights
        debug_J = nnCostFunction(self.nn_params, self.n_in, self.n_hidden,
                                 self.n_out, self.X, self.Y, lambda_value=3)
        print(f"Cost (w/ lambda = 10):\t {debug_J[0][0]} (expected 0.576051)")
        return

    def train(self, max_iter, lambda_value, debug=False):

        costFunction: Callable[[Any], tuple[
            Union[float, Any], ndarray]] = lambda p: nnCostFunction(
            p, self.n_in, self.n_hidden, self.n_out,
            self.X, self.Y, lambda_value
        )

        [nn_params, cost] = fmincg(costFunction, self.nn_params, max_iter, debug)

        self.t1 = np.reshape(nn_params[0:self.n_hidden * (self.n_in + 1)],
                             (self.n_hidden, (self.n_in + 1)), order='F')
        self.t2 = np.reshape(nn_params[(self.n_hidden * (self.n_in + 1)):],
                             (self.n_out, (self.n_hidden + 1)), order='F')

        self.cost.append(cost)
        return self


network = ANN(400, 25, 10, True)
network.reset()
m = scipy.io.loadmat('../data/digitdata.mat')
x = m['X']
y = m['y']
y = np.squeeze(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.34, random_state=10)
network.set_data(x_train, y_train)
network.train(50, 3, True)
print(network.prediction_accuracy())
