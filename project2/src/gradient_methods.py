# author: Markus Bjørklund and Magnus Kristoffersen
# creation date: 2023-11-06
# encoding: utf-8
""" Gradient methods """


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import autograd.numpy as np
from autograd import grad


def sigmoid(X, beta):
    return 1 / (1 + np.exp(-X @ beta))


def cost_OLS(X, y, beta):
    """Cost function for Ordinary Least Squares"""
    val = np.sum((y - X @ beta) ** 2)
    return val


def cost_ridge(X, y, beta, lmb):
    """Cost function for Ridge regression"""
    val = np.sum((y - X @ beta) ** 2 + lmb * np.linalg.norm(beta))
    return val


def cost_logistic(X, y, beta, lmb=0):
    """Cost function for Logistic regression"""
    delta = 1e-14
    prob = sigmoid(X, beta)
    val = -(np.dot(y, np.log(prob + delta)) + np.dot((1 - y), np.log(1 - prob + delta))) + lmb * np.linalg.norm(beta)
    return val


def gradient_logistic(beta_log, n, X, y, lmb=0):
    gradient = (1 / n) * X.T @ (sigmoid(X, beta_log) - y) + lmb * beta_log
    return gradient


def gradient_OLS(beta_OLS, n, X, y):
    """Analytical gradient for OLS"""
    gradient = (2.0 / n) * X.T @ (X @ beta_OLS - y)
    return gradient


def gradient_ridge(beta_ridge, n, X, y, lmb):
    """Analytical gradient for Ridge regression"""
    gradient_ridge = 2 * (1 / n * X.T @ (X @ beta_ridge - y) + lmb * beta_ridge)
    return gradient_ridge


def step_length(t, t0, t1):
    """Variable learning rate (time decay rate)"""
    return t0 / (t + t1)


##################
# Gradient methods
##################


def GD_plain(
    X,
    y,
    n,
    p,
    type_=None,
    grad_type="an",
    lmb=0,
    eta=0.001,
    arg_seed=None,
    max_iter=2.5e4,
):
    """Plain gradient descent"""

    np.random.seed(arg_seed)
    beta = np.random.randn(
        p, 1
    )  # Initialize random beta, p is dim+1 (p=3 for a second order polynomial)

    eps = 1e-8  # Stopping criterion
    counter = 0
    gradient = 100  # Random initializing of gradient because of while loop

    auto_grad_OLS = grad(cost_OLS, 2)
    auto_grad_ridge = grad(cost_ridge, 2)
    auto_grad_log = grad(cost_logistic, 2)

    beta_per_iter = [list(np.ravel(beta))]  # List because dynamic size

    while np.linalg.norm(gradient) > eps and counter < max_iter and counter < max_iter:
        # Calculating gradient analytically
        if type_ == "OLS":
            if grad_type == "an":
                gradient = gradient_OLS(beta, n, X, y)
            elif grad_type == "auto":
                gradient = 1 / n * auto_grad_OLS(X, y, beta)

        elif type_ == "ridge":
            if grad_type == "an":
                gradient = gradient_ridge(beta, n, X, y, lmb)
            elif grad_type == "auto":
                gradient = 1 / n * auto_grad_ridge(X, y, beta, lmb)

        elif type_ == "log":
            if grad_type == "an":
                gradient = gradient_logistic(beta, n, X, y, lmb)
            elif grad_type == "auto":
                gradient = 1 / n * auto_grad_log(X, y, beta, lmb)

        # Updating beta
        beta -= eta * gradient
        counter += 1

        # Weird bug. List filled up with pointers to beta (which updated the whole list every time beta changed)
        # instead of the actual values for the current iteration. Have to make nested list instead.
        beta_per_iter.append(list(np.ravel(beta)))

    print("Done after {} iterations".format(counter))
    return beta, counter, beta_per_iter


def GD_momentum(
    X,
    y,
    n,
    p,
    type_=None,
    grad_type="an",
    lmb=0,
    eta=0.001,
    gamma=0.5,
    arg_seed=None,
    max_iter=2.5e4,
):
    """Gradient descent with momentum"""

    np.random.seed(arg_seed)
    beta = np.random.randn(p, 1)  # Initialize random beta
    eps = 1e-8  # Stopping criterion
    counter = 0
    gradient = 100  # Random initializing of gradient because of while loop
    v = [0]  # Initialize with zero for first iteration

    auto_grad_OLS = grad(cost_OLS, 2)
    auto_grad_ridge = grad(cost_ridge, 2)
    auto_grad_log = grad(cost_logistic, 2)

    beta_per_iter = [list(np.ravel(beta))]  # List because dynamic size

    while np.linalg.norm(gradient) > eps and counter < max_iter:
        # Calculating gradient analytically
        if type_ == "OLS":
            if grad_type == "an":
                gradient = gradient_OLS(beta, n, X, y)
            elif grad_type == "auto":
                gradient = 1 / n * auto_grad_OLS(X, y, beta)

        elif type_ == "ridge":
            if grad_type == "an":
                gradient = gradient_ridge(beta, n, X, y, lmb)
            elif grad_type == "auto":
                gradient = 1 / n * auto_grad_ridge(X, y, beta, lmb)

        elif type_ == "log":
            if grad_type == "an":
                gradient = gradient_logistic(beta, n, X, y, lmb)
            elif grad_type == "auto":
                gradient = 1 / n * auto_grad_log(X, y, beta, lmb)

        mom = gamma * v[counter] + eta * gradient
        v.append(mom)

        # Updating beta
        beta -= mom
        counter += 1

        beta_per_iter.append(list(np.ravel(beta)))

    print("Done after {} iterations".format(counter))
    return beta, counter, beta_per_iter


def SGD(
    X,
    y,
    n,
    p,
    type_=None,
    grad_type="an",
    epochs=3000,
    M=5,
    eta=None,
    t0=200,
    t1=1500,
    lmb=0,
    arg_seed=None,
):
    """Plain stochastic gradient descent for OLS"""

    np.random.seed(arg_seed)
    beta = np.random.randn(p, 1)  # Initialize random beta
    m = int(n / M)  # Number of minibatches

    auto_grad_OLS = grad(cost_OLS, 2)
    auto_grad_ridge = grad(cost_ridge, 2)
    auto_grad_log = grad(cost_logistic, 2)

    beta_per_iter = [list(np.ravel(beta))]  # List because dynamic size

    for e in range(epochs):
        for i in range(m):
            # Pick random batch number K, use only data from that batch
            k = np.random.randint(m)
            batch_start = k * M
            batch_end = batch_start + M
            X_i = X[batch_start:batch_end]
            y_i = y[batch_start:batch_end]

            # Calculating gradient analytically
            if type_ == "OLS":
                if grad_type == "an":
                    gradient = gradient_OLS(beta, M, X_i, y_i)
                elif grad_type == "auto":
                    gradient = 1 / M * auto_grad_OLS(X_i, y_i, beta)

            elif type_ == "ridge":
                if grad_type == "an":
                    gradient = gradient_ridge(beta, M, X_i, y_i, lmb)
                elif grad_type == "auto":
                    gradient = 1 / M * auto_grad_ridge(X_i, y_i, beta, lmb)

            elif type_ == "log":
                if grad_type == "an":
                    gradient = gradient_logistic(beta, M, X_i, y_i, lmb)
                elif grad_type == "auto":
                    gradient = 1 / M * auto_grad_log(X_i, y_i, beta, lmb)

            if eta:
                # Constant learning rate
                gamma_j = eta
            else:
                # Calculate variable learning rate
                t = e * m + i
                gamma_j = step_length(t, t0, t1)

            # Updating beta
            beta -= gamma_j * gradient

        beta_per_iter.append(
            list(np.ravel(beta))
        )  # Save beta per epoch (not per batch)

    # print("gamma_j after %d epochs: %g" % (epochs,gamma_j))
    return beta, beta_per_iter


def SGD_momentum(
    X,
    y,
    n,
    p,
    type_=None,
    grad_type="an",
    epochs=3000,
    M=5,
    eta=None,
    t0=200,
    t1=1500,
    lmb=0,
    gamma=0.5,
    arg_seed=None,
):
    """Plain stochastic gradient descent for OLS"""

    np.random.seed(arg_seed)
    beta = np.random.randn(p, 1)  # Initialize random beta
    m = int(n / M)  # Number of minibatches

    v = [0]
    counter = 0

    auto_grad_OLS = grad(cost_OLS, 2)
    auto_grad_ridge = grad(cost_ridge, 2)
    auto_grad_log = grad(cost_logistic, 2)

    beta_per_iter = [list(np.ravel(beta))]  # List because dynamic size

    for e in range(epochs):
        for i in range(m):
            # Pick random batch number K, use only data from that batch
            k = np.random.randint(m)
            batch_start = k * M
            batch_end = batch_start + M
            X_i = X[batch_start:batch_end]
            y_i = y[batch_start:batch_end]

            # Calculating gradient analytically
            if type_ == "OLS":
                if grad_type == "an":
                    gradient = gradient_OLS(beta, M, X_i, y_i)
                elif grad_type == "auto":
                    gradient = 1 / M * auto_grad_OLS(X_i, y_i, beta)

            elif type_ == "ridge":
                if grad_type == "an":
                    gradient = gradient_ridge(beta, M, X_i, y_i, lmb)
                elif grad_type == "auto":
                    gradient = 1 / M * auto_grad_ridge(X_i, y_i, beta, lmb)

            elif type_ == "log":
                if grad_type == "an":
                    gradient = gradient_logistic(beta, M, X_i, y_i, lmb)
                elif grad_type == "auto":
                    gradient = 1 / M * auto_grad_log(X_i, y_i, beta, lmb)

            if eta:
                # Constant learning rate
                gamma_j = eta
            else:
                # Calculate variable learning rate
                t = e * m + i
                gamma_j = step_length(t, t0, t1)

            # Calculate momentum
            mom = gamma * v[counter] + gamma_j * gradient
            v.append(mom)

            # Updating beta
            beta -= mom

            counter += 1

        beta_per_iter.append(
            list(np.ravel(beta))
        )  # Save beta per epoch (not per batch)

    # print("gamma_j after %d epochs: %g" % (epochs,gamma_j))
    return beta, beta_per_iter


def AdaGrad_GD(
    X,
    y,
    n,
    p,
    type_=None,
    grad_type="an",
    lmb=0,
    eta=0.001,
    arg_seed=None,
    max_iter=2.5e4,
):
    """AdaGrad for plain gradient descent"""

    np.random.seed(arg_seed)
    beta = np.random.randn(p, 1)  # Initialize random beta
    eps = 1e-8  # Stopping criterion
    counter = 0
    gradient = 100  # Random initializing of gradient because of while loop
    delta = 1e-7  # Small constant for numerical stability
    r = 0  # Gradient accumulation variable

    auto_grad_OLS = grad(cost_OLS, 2)
    auto_grad_ridge = grad(cost_ridge, 2)
    auto_grad_log = grad(cost_logistic, 2)

    beta_per_iter = [list(np.ravel(beta))]  # List because dynamic size

    while np.linalg.norm(gradient) > eps and counter < max_iter:
        # Calculating gradient analytically
        if type_ == "OLS":
            if grad_type == "an":
                gradient = gradient_OLS(beta, n, X, y)
            elif grad_type == "auto":
                gradient = 1 / n * auto_grad_OLS(X, y, beta)

        elif type_ == "ridge":
            if grad_type == "an":
                gradient = gradient_ridge(beta, n, X, y, lmb)
            elif grad_type == "auto":
                gradient = 1 / n * auto_grad_ridge(X, y, beta, lmb)

        elif type_ == "log":
            if grad_type == "an":
                gradient = gradient_logistic(beta, n, X, y, lmb)
            elif grad_type == "auto":
                gradient = 1 / n * auto_grad_log(X, y, beta, lmb)

        # Accumulate gradient
        r += gradient * gradient  # Elementwise multiplication (Hadamard)

        # Updating beta
        beta -= eta / (delta + np.sqrt(r)) * gradient

        counter += 1

        beta_per_iter.append(list(np.ravel(beta)))  # Save beta per iteration

        # if counter % 100000 == 0:
        # print("Iteration: {}".format(counter))
    print("ADA done after {} iterations".format(counter))
    return beta, counter, beta_per_iter


def AdaGrad_GD_mom(
    X,
    y,
    n,
    p,
    type_=None,
    grad_type="an",
    lmb=0,
    eta=0.01,
    gamma=0.5,
    arg_seed=None,
    max_iter=2.5e4,
):
    """ADA for GD with momentum"""

    np.random.seed(arg_seed)
    beta = np.random.randn(p, 1)  # Initialize random beta
    eps = 1e-8  # Stopping criterion
    counter = 0
    gradient = 100  # Random initializing of gradient because of while loop
    v = [0]  # Initialize with zero for first iteration

    delta = 1e-7  # Small constant for numerical stability
    r = 0  # Gradient accumulation variable

    auto_grad_OLS = grad(cost_OLS, 2)
    auto_grad_ridge = grad(cost_ridge, 2)
    auto_grad_log = grad(cost_logistic, 2)

    beta_per_iter = [list(np.ravel(beta))]  # List because dynamic size

    while np.linalg.norm(gradient) > eps and counter < max_iter:
        # Calculating gradient analytically
        if type_ == "OLS":
            if grad_type == "an":
                gradient = gradient_OLS(beta, n, X, y)
            elif grad_type == "auto":
                gradient = 1 / n * auto_grad_OLS(X, y, beta)

        elif type_ == "ridge":
            if grad_type == "an":
                gradient = gradient_ridge(beta, n, X, y, lmb)
            elif grad_type == "auto":
                gradient = 1 / n * auto_grad_ridge(X, y, beta, lmb)

        elif type_ == "log":
            if grad_type == "an":
                gradient = gradient_logistic(beta, n, X, y, lmb)
            elif grad_type == "auto":
                gradient = 1 / n * auto_grad_log(X, y, beta, lmb)

        # Accumulate gradient
        r += gradient * gradient  # Elementwise multiplication (Hadamard)

        # Calculating momentum term, with variable learning rate
        mom = gamma * v[counter] + eta / (delta + np.sqrt(r)) * gradient
        v.append(mom)

        # Updating beta
        beta -= mom

        counter += 1

        beta_per_iter.append(list(np.ravel(beta)))  # Save beta per iteration

    print("ADA with momentum done after {} iterations".format(counter))
    return beta, counter, beta_per_iter


def AdaGrad_SGD(
    X,
    y,
    n,
    p,
    type_=None,
    grad_type="an",
    epochs=4000,
    M=5,
    lmb=0,
    eta=0.1,
    arg_seed=None,
):
    """ADA with SGD for OLS"""

    np.random.seed(arg_seed)
    beta = np.random.randn(p, 1)  # Initialize random beta
    m = int(n / M)  # Number of minibatches
    delta = 1e-7  # Small constant for numerical stability
    r = 0  # Gradient accumulation variable

    auto_grad_OLS = grad(cost_OLS, 2)
    auto_grad_ridge = grad(cost_ridge, 2)
    auto_grad_log = grad(cost_logistic, 2)

    beta_per_iter = [list(np.ravel(beta))]  # List because dynamic size

    for e in range(epochs):
        for i in range(m):
            # Pick random batch number K, use only data from that batch
            k = np.random.randint(m)
            batch_start = k * M
            batch_end = batch_start + M
            X_i = X[batch_start:batch_end]
            y_i = y[batch_start:batch_end]

            # Calculating gradient analytically
            if type_ == "OLS":
                if grad_type == "an":
                    gradient = gradient_OLS(beta, M, X_i, y_i)
                elif grad_type == "auto":
                    gradient = 1 / M * auto_grad_OLS(X_i, y_i, beta)

            elif type_ == "ridge":
                if grad_type == "an":
                    gradient = gradient_ridge(beta, M, X_i, y_i, lmb)
                elif grad_type == "auto":
                    gradient = 1 / M * auto_grad_ridge(X_i, y_i, beta, lmb)

            elif type_ == "log":
                if grad_type == "an":
                    gradient = gradient_logistic(beta, M, X_i, y_i, lmb)
                elif grad_type == "auto":
                    gradient = 1 / M * auto_grad_log(X_i, y_i, beta, lmb)

            # Accumulate gradient
            r += gradient * gradient  # Elementwise multiplication (Hadamard)

            # Updating beta
            beta -= eta / (delta + np.sqrt(r)) * gradient

        beta_per_iter.append(
            list(np.ravel(beta))
        )  # Save beta per epoch (not per batch)

    print(
        "learning rate after {} epochs: {}".format(epochs, eta / (delta + np.sqrt(r)))
    )
    return beta, beta_per_iter


def AdaGrad_SGD_mom(
    X,
    y,
    n,
    p,
    type_="ridge",
    grad_type="an",
    epochs=4000,
    M=5,
    lmb=0,
    eta=0.1,
    gamma=0.5,
    arg_seed=None,
):
    """ADA with SGD and momentum for OLS"""

    np.random.seed(arg_seed)
    beta = np.random.randn(p, 1)  # Initialize random beta
    m = int(n / M)  # Number of minibatches
    delta = 1e-7  # Small constant for numerical stability
    r = 0  # Gradient accumulation variable

    v = [0]
    counter = 0

    auto_grad_OLS = grad(cost_OLS, 2)
    auto_grad_ridge = grad(cost_ridge, 2)
    auto_grad_log = grad(cost_logistic, 2)

    beta_per_iter = [list(np.ravel(beta))]  # List because dynamic size

    for e in range(epochs):
        for i in range(m):
            # Pick random batch number K, use only data from that batch
            k = np.random.randint(m)
            batch_start = k * M
            batch_end = batch_start + M
            X_i = X[batch_start:batch_end]
            y_i = y[batch_start:batch_end]

            # Calculating gradient analytically
            if type_ == "OLS":
                if grad_type == "an":
                    gradient = gradient_OLS(beta, M, X_i, y_i)
                elif grad_type == "auto":
                    gradient = 1 / M * auto_grad_OLS(X_i, y_i, beta)

            elif type_ == "ridge":
                if grad_type == "an":
                    gradient = gradient_ridge(beta, M, X_i, y_i, lmb)
                elif grad_type == "auto":
                    gradient = 1 / M * auto_grad_ridge(X_i, y_i, beta, lmb)

            elif type_ == "log":
                if grad_type == "an":
                    gradient = gradient_logistic(beta, M, X_i, y_i, lmb)
                elif grad_type == "auto":
                    gradient = 1 / M * auto_grad_log(X_i, y_i, beta, lmb)

            # Accumulate gradient
            r += gradient * gradient  # Elementwise multiplication (Hadamard)

            # Calculating momentum term, with variable learning rate
            mom = gamma * v[counter] + eta / (delta + np.sqrt(r)) * gradient
            v.append(mom)

            # Updating beta
            beta -= mom

            counter += 1

        beta_per_iter.append(
            list(np.ravel(beta))
        )  # Save beta per epoch (not per batch)

    print(
        "learning rate after {} epochs: {}".format(epochs, eta / (delta + np.sqrt(r)))
    )
    return beta, beta_per_iter


def RMSProp_SGD(
    X,
    y,
    n,
    p,
    type_=None,
    grad_type="an",
    epochs=4000,
    M=5,
    lmb=0,
    eta=0.01,
    rho=0.99,
    arg_seed=None,
):
    """RMSProp with SGD"""

    np.random.seed(arg_seed)
    beta = np.random.randn(p, 1)  # Initialize random beta
    m = int(n / M)  # Number of minibatches
    delta = 1e-6  # Small constant for numerical stability
    r = 0  # Gradient accumulation variable

    auto_grad_OLS = grad(cost_OLS, 2)
    auto_grad_ridge = grad(cost_ridge, 2)
    auto_grad_log = grad(cost_logistic, 2)

    beta_per_iter = [list(np.ravel(beta))]  # List because dynamic size

    for e in range(epochs):
        for i in range(m):
            # Pick random batch number K, use only data from that batch
            k = np.random.randint(m)
            batch_start = k * M
            batch_end = batch_start + M
            X_i = X[batch_start:batch_end]
            y_i = y[batch_start:batch_end]

            # Calculating gradient analytically
            if type_ == "OLS":
                if grad_type == "an":
                    gradient = gradient_OLS(beta, M, X_i, y_i)
                elif grad_type == "auto":
                    gradient = 1 / M * auto_grad_OLS(X_i, y_i, beta)

            elif type_ == "ridge":
                if grad_type == "an":
                    gradient = gradient_ridge(beta, M, X_i, y_i, lmb)
                elif grad_type == "auto":
                    gradient = 1 / M * auto_grad_ridge(X_i, y_i, beta, lmb)

            elif type_ == "log":
                if grad_type == "an":
                    gradient = gradient_logistic(beta, M, X_i, y_i, lmb)
                elif grad_type == "auto":
                    gradient = 1 / M * auto_grad_log(X_i, y_i, beta, lmb)

            # Accumulate gradient
            gradient_2 = gradient * gradient  # Elementwise multiplication (Hadamard)
            r = rho * r + (1 - rho) * gradient_2

            # Updating beta
            beta -= (eta / np.sqrt(delta + r)) * gradient

        beta_per_iter.append(
            list(np.ravel(beta))
        )  # Save beta per epoch (not per batch)

    print("learning rate after {} epochs: {}".format(epochs, eta / np.sqrt(delta + r)))
    return beta, beta_per_iter


def ADAM_SGD(
    X,
    y,
    n,
    p,
    type_=None,
    grad_type="an",
    epochs=4000,
    M=5,
    lmb=0,
    eta=0.1,
    rho_1=0.9,
    rho_2=0.999,
    arg_seed=None,
):
    """ADAM with SGD for OLS"""

    np.random.seed(arg_seed)
    beta = np.random.randn(p, 1)  # Initialize random beta
    m = int(n / M)  # Number of minibatches
    delta = 1e-8  # Small constant for numerical stability

    t = 0  # Time step

    auto_grad_OLS = grad(cost_OLS, 2)
    auto_grad_ridge = grad(cost_ridge, 2)
    auto_grad_log = grad(cost_logistic, 2)

    beta_per_iter = [list(np.ravel(beta))]  # List because dynamic size

    for e in range(epochs):
        s = 0  # First moment variable
        r = 0  # Second moment variable

        # Updating time step
        t += 1

        for i in range(m):
            # Pick random batch number K, use only data from that batch
            k = np.random.randint(m)
            batch_start = k * M
            batch_end = batch_start + M
            X_i = X[batch_start:batch_end]
            y_i = y[batch_start:batch_end]

            # Calculating gradient analytically
            if type_ == "OLS":
                if grad_type == "an":
                    gradient = gradient_OLS(beta, M, X_i, y_i)
                elif grad_type == "auto":
                    gradient = 1 / M * auto_grad_OLS(X_i, y_i, beta)

            elif type_ == "ridge":
                if grad_type == "an":
                    gradient = gradient_ridge(beta, M, X_i, y_i, lmb)
                elif grad_type == "auto":
                    gradient = 1 / M * auto_grad_ridge(X_i, y_i, beta, lmb)

            elif type_ == "log":
                if grad_type == "an":
                    gradient = gradient_logistic(beta, M, X_i, y_i, lmb)
                elif grad_type == "auto":
                    gradient = 1 / M * auto_grad_log(X_i, y_i, beta, lmb)

            # Update biased first moment
            s = rho_1 * s + (1 - rho_1) * gradient

            # Update biased second moment
            r = rho_2 * r + (1 - rho_2) * gradient * gradient

            # Correct bias in first moment
            s = s / (1 - rho_1**t)

            # Correct bias in second moment
            r = r / (1 - rho_2**t)

            # Updating beta
            beta -= eta * (s / (np.sqrt(r) + delta))

        beta_per_iter.append(
            list(np.ravel(beta))
        )  # Save beta per epoch (not per batch)

    print("learning rate after {} epochs: {}".format(epochs, eta / np.sqrt(delta + r)))
    return beta, beta_per_iter
