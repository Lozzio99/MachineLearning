from typing import Callable, Any, List

from sigmoid import sigmoid


def iterate(X, f):
    return [f(xi) for xi in X]


derivative: Callable[[Any], Any] = lambda g: g * (1 - g)
g_float: Callable[[float], float] = lambda Z: sigmoid(Z)
g_list: Callable[
    [list[float]], list[float]
] = lambda Z: [iterate(Xi, g_float) for Xi in Z] if type(Z[0]) == list else iterate(Z, g_float)


def sigmoidGradient(z):
    # SIGMOIDGRADIENT returns the gradient of the sigmoid function
    # evaluated at z
    #   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
    #   evaluated at z. This should work regardless if z is a matrix or a
    #   vector. In particular, if z is a vector or matrix, you should return
    #   the gradient for each element.

    # The value g should be correctly computed by your code below.

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of the sigmoid function evaluated at
    #               each value of z (z can be a matrix, vector or scalar).

    # =============================================================

    t = type(z)
    if t == int:
        return derivative(g_float(z))
    elif t == float:
        return derivative(g_float(z))
    elif t == list:
        return derivative(g_list(z))
    else:
        return derivative(sigmoid(z))
