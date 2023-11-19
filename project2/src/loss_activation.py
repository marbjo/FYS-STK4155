# author: Markus Bjørklund and Magnus Kristoffersen
# creation date: 2023-11-06
# encoding: utf-8
""" Loss and activation functions for neural net """

import numpy as np


def linear(x):
    return x


def linear_deriv(x):
    return 1


def sigmoid(x):
    """Activation function"""
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    """Derivative of sigmoid activation function"""
    return sigmoid(x) * (1 - sigmoid(x))


def RELU(x):
    return np.where(x > np.zeros(x.shape), x, np.zeros(x.shape))


def RELU_deriv(x):
    return np.where(x > 0, 1, 0)


def leaky_RELU(alpha):
    def func(x):
        return np.where(x > 0, x, alpha * x)
    return func


def leaky_RELU_deriv(alpha):
    def func(x):
        return np.where(x > 0, 1, alpha)
    return func


"""Every loss function has to be defined with the same argument structure (x, target)"""


def cross_entropy_loss(x, target):
    delta = 1e-14  # For numerical stability
    N = input.size(target)
    return -(1 / N) * np.sum(target * np.log(x + delta), axis=1, keepdims=True)


def cross_entropy_loss_deriv(x, target):
    delta = 1e-14  # For numerical stability

    return (1 - target) / (1 - x + delta) - target / (x + delta)


def mse_loss(x, target):
    return np.sum((target - x) ** 2, keepdims=True) / len(target)


def mse_loss_deriv(x, target):
    return 2 * np.sum(x - target, axis=1, keepdims=True) / len(target)


def l2_regularization_deriv(lambda_val, weight):
    return lambda_val * weight
