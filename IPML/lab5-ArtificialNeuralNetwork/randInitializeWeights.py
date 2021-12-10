import numpy as np


def randInitializeWeights(L_in, L_out):
    # RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
    # incoming connections and L_out outgoing connections
    #   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights
    #   of a layer with L_in incoming connections and L_out outgoing
    #   connections.
    #
    #   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
    #   the first row of W handles the "bias" terms
    #

    # You need to return the following variables correctly
    W = np.zeros((L_out, 1 + L_in))
    epsilon = 0.12
    for i in range(0, L_out):
        for k in range(0, L_in + 1):
            if i == 0:
                W[i, k] = 1
            else:
                W[i, k] = np.random.uniform(low=-epsilon, high=epsilon)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Initialize W randomly so that we break the symmetry while
    #               training the neural network.
    #
    # Note: The first row of W corresponds to the parameters for the bias units
    #

    # =========================================================================

    return W
