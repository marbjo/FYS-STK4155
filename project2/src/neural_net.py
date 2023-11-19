# author: Markus Bjørklund and Magnus Kristoffersen
# creation date: 2023-11-06
# encoding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from autograd import grad

# from sklearn.neural_network import MLPClassifier

class Scheduler:
    def __init__(self):
        pass

    def update_learning(self, grad):
        pass

    def reset(self):
        pass

class Fixed:
    def __init__(self, eta, mom=0):
        self.eta = eta
        self.mom = mom
        self.change = 0.

    def update_learning(self, grad):
        self.change = self.eta * grad + self.mom * self.change
        return self.change
    
    def reset(self):
        self.change = 0.


class Adagrad:
    def __init__(self, eta, mom=0):
        self.acc_sqr_grad = None
        self.eta = eta
        self.mom = mom
        self.change = 0.

    def update_learning(self, grad):
        delta = 1e-14

        if self.acc_sqr_grad == None:
            self.acc_sqr_grad = np.zeros(np.shape(grad))
        
        self.acc_sqr_grad += grad * grad

        self.change = self.eta * grad / (np.sqrt(self.acc_sqr_grad) + delta) + self.mom * self.change

        return self.change

    def reset(self):
        self.acc_sqr_grad = None
        self.change = 0.


class RMSProp:
    def __init__(self, eta, decay_second = 0.99):
        self.acc_sqr_grad = None
        self.eta = eta
        self.decay_second = decay_second

    def update_learning(self, grad):
        delta = 1e-14

        if self.acc_sqr_grad == None:
            self.acc_sqr_grad = np.zeros(np.shape(grad))
        
        self.acc_sqr_grad = self.decay_second * self.acc_sqr_grad + (1 - self.decay_second) * grad * grad

        return self.eta * grad / (np.sqrt(self.acc_sqr_grad) + delta)
    
    def reset(self):
        self.acc_sqr_grad = None


class ADAM:
    def __init__(self, eta, decay_first = 0.9, decay_second = 0.999):
        self.acc_grad = None
        self.acc_sqr_grad = None
        self.eta = eta
        self.decay_first = decay_first
        self.decay_second = decay_second
        self.epoch = 1

    def update_learning(self, grad):
        delta = 1e-14

        if self.acc_grad == None:
            self.acc_grad = np.zeros(np.shape(grad))

        if self.acc_sqr_grad == None:
            self.acc_sqr_grad = np.zeros(np.shape(grad))
        
        self.acc_grad = self.decay_first * self.acc_grad + (1 - self.decay_first) * grad
        self.acc_sqr_grad = self.decay_second * self.acc_sqr_grad + (1 - self.decay_second) * grad * grad

        first_term = self.acc_grad / (1 - self.decay_first ** (self.epoch))
        second_term = self.acc_sqr_grad / (1 - self.decay_second ** (self.epoch))

        return self.eta * first_term / (np.sqrt(second_term) + delta)
    
    def reset(self):
        self.acc_grad = None
        self.acc_sqr_grad = None
        self.epoch += 1


class Layer:
    def __init__(self, input_n, output_n, weight_schedule, bias_schedule, arg_seed=None):
        """Create random weights and (zero) bias for a chosen number of input/output"""

        # Initializing weights (randomly, with seed keyword argument) and bias
        np.random.seed(arg_seed)
        self.w = np.random.randn(input_n, output_n)
        self.b = np.zeros(output_n) + 0.01 # Common to initalize bias as zero, or very small number (not normal distribution)
        self.eta_w_schedule = weight_schedule
        self.eta_b_schedule = bias_schedule

    def feed_forward(self, X):
        """Feed forward step"""
        self.X = X
        return self.X @ self.w + self.b  # XW + b

    def backward_prop(self, dE_dY, reg_deriv_val):
        """Backpropagation step"""

        # Compute the derivative of the error wrt the weights
        dE_dW = self.X.T @ dE_dY + reg_deriv_val
        # Compute the derivative of the error wrt the input
        dE_dX = dE_dY @ self.w.T

        # Gradient descent for weights and bias
        self.w -= self.eta_w_schedule.update_learning(dE_dW)
        self.b -= self.eta_b_schedule.update_learning(np.sum(dE_dY, axis=0))
        # Return the derivative wrt to the input, as it serves as dE_dY for the next (earlier) layer
        return dE_dX


class Activation_Layer:
    def __init__(self, func, deriv_func):
        self.func = func
        self.deriv_func = deriv_func
        self.w = 0
        self.b = 0
        self.eta_w_schedule = Scheduler()
        self.eta_b_schedule = Scheduler()

    def feed_forward(self, X):
        self.X = X
        return self.func(self.X)

    def backward_prop(self, dE_dY, reg_deriv_val):
        """
        Has extra (unused) argument eta because every layer should be able to call forward/backward
        methods the same way.
        """
        return np.multiply(dE_dY, self.deriv_func(self.X))


def train(
    net,
    X,
    Y,
    loss_func_deriv,
    regularization_deriv,
    batches=1,
    epoch_n=10000,
    lambda_val=0,
    grad_type="an",
):
    """Function for training the network. Gradient descent is decided by derivative of loss func"""
    batch_size = len(Y) // batches
    for epochs in range(epoch_n):
        rearrange = np.arange(len(Y))
        np.random.shuffle(rearrange)

        X = X[rearrange]
        Y = Y[rearrange]

        for batch in range(batches):
            output = X[batch*batch_size:(batch_size+1)*batch_size]
            target = Y[batch*batch_size:(batch_size+1)*batch_size]

            for layer in net:
                # Feed forward through the layers. Input for new layer is output from the earlier one
                output = layer.feed_forward(output)

            # Backpropagation. Last layer gets gradient from loss function
            grad = loss_func_deriv(output, target)

            for layer in reversed(net):
                # Subsequent layers gets gradient from the input of previous layer
                reg_deriv_value = regularization_deriv(lambda_val, layer.w)
                grad = layer.backward_prop(grad, reg_deriv_value)
        
        for layer in net:
            layer.eta_w_schedule.reset()
            layer.eta_b_schedule.reset()


def predict(net, X, Y):
    """One full feed-forward pass"""
    test_list = []
    output = X
    for layer in net:
        output = layer.feed_forward(output)

    test_list.append(output)
    return test_list


def predict_gate(net, X, Y):
    """Prediction"""
    for x, y in zip(X, Y):
        output = x
        for layer in net:
            output = layer.feed_forward(output)

        print("Input: ({},{}) gives output {}".format(x[0], x[1], output))


def accuracy(prediction, target):
    """Accuracy score for classification"""
    if len(prediction) != len(target):
        raise Exception("Length of predictions and targets does not match")
    return np.sum(np.equal(prediction, target)) / len(prediction)
