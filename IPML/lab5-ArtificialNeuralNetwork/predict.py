import numpy as np
from numpy import shape

from sigmoid import sigmoid


def predict(Theta1, Theta2, X):
    # PREDICT Predict the label of an input given a trained neural network
    #   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    #   trained weights of a neural network (Theta1, Theta2)

    # Useful values
    m = np.shape(X)[0]  # number of examples
    # You need to return the following variables correctly
    p = np.zeros(m)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned neural network. You should set p to a
    #               vector containing labels between 1 to num_labels.
    #
    a0 = np.dot(np.hstack((np.ones((m, 1)), X)),  np.transpose(Theta1))
    a1 = np.dot(np.hstack((np.ones((m, 1)), sigmoid(a0))), np.transpose(Theta2))
    a1 = sigmoid(a1)

    for i in range(0, m):
        p[i] = (np.argmax(a1[i], axis=0) + 1)

    return p

    # X (5000, 400)
    # i  (1, 400)
    # T1 (25, 400+1)
    # T2 (10, 25+1)

    # accuracy 96.5 with this :

    # for i in range(0, m):
    #    instance = X[i]
    #    a1 = multiply(instance, Theta1)
    #    a2 = multiply(a1, Theta2)
    #    p[i] = (np.argmax(a2, axis=0) + 1)
    # return p


# =========================================================================


def multiply(N, M):
    n_ = np.insert(N, 0, 1)             # bias
    m = shape(M)[0]                     # theta_i
    res = np.zeros(m)
    for i in range(0, m):               # for each theta_i
        s = 0
        for j in range(0, len(N)):      # sum (theta_i_j * x_j) for each j
            s += M[i, j] * n_[j]
        res[i] = sigmoid(s)

    return res
