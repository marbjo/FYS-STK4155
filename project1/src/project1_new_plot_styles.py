# authors: Magnus Kristoffersen and Markus Bjørklund
# creation date: 2022-09-06
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from imageio import imread

import os
import sys

plt.style.use("seaborn-whitegrid")
matplotlib.rc("font", size=20)


def franke_func(x, y, epsilon=0, rng=np.random.default_rng(seed=0)):
    """
    Function for calculating the value of the Franke function in the point (x, y), 
    with an added normal distributed noise term.
    """

    term_1 = 3 / 4 * np.exp(-((9 * x - 2) ** 2) / 4 - (9 * y - 2) ** 2 / 4)
    term_2 = 3 / 4 * np.exp(-((9 * x + 1) ** 2) / 49 - (9 * y + 1) / 10)
    term_3 = 1 / 2 * np.exp(-((9 * x - 7) ** 2) / 4 - (9 * y - 3) ** 2 / 4)
    term_4 = -1 / 5 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    noise_term = epsilon * rng.normal(size=len(x))

    return term_1 + term_2 + term_3 + term_4 + noise_term


def design_matrix(x, y, degree):
    """
    Function for generating the design matrix for a 2-dimensional polynomial.
    The size of the design matrix is decided by the parameter 'degree', 
    which gives a column vector in the design matrix for all combinations 
    of x^i y^j where 0 < i + j <= degree.
    
    The function returns the design matrix and the mean values of each column in the design matrix.
    """

    num_cols = np.sum([deg + 1 for deg in range(1, degree + 1)])

    X = np.zeros((len(x), num_cols))
    X_offset = np.zeros(num_cols)

    # Create the unscaled design matrix and calculating the scaling parameters
    col = 0
    for i in range(1, degree + 1):
        for j in range(i + 1):
            X[:, col] = x ** (i - j) * y ** (j)
            X_offset[col] = np.mean(x ** (i - j) * y ** (j))

            col += 1

    return X, X_offset


def beta_ols(X, f_xy):
    """
    Function for calculating the beta parameters for a linear regression model 
    using ordinary least squares. To handle the possibility of the matrix transpose(X)*X being non-invertible,
    we use singular value decomposition to calculate the pseudoinverse.
    """

    U, s, V_T = np.linalg.svd(np.matmul(np.transpose(X), X))
    inv_mat = np.matmul(np.matmul(np.transpose(V_T), np.diag(1 / s)), np.transpose(U))
    return np.matmul(np.matmul(inv_mat, np.transpose(X)), f_xy)


def beta_ridge(X, f_xy, lmb):
    """
    Function for calculating the beta parameters for a linear regression model 
    using ridge regression. To handle the possibility of the matrix transpose(X)*X being non-invertible,
    we use singular value decomposition to calculate the pseudoinverse.
    """

    U, s, V_T = np.linalg.svd(np.matmul(np.transpose(X), X) + lmb * np.eye(len(X[0])))
    inv_mat = np.matmul(np.matmul(np.transpose(V_T), np.diag(1 / s)), np.transpose(U))
    return np.matmul(np.matmul(inv_mat, np.transpose(X)), f_xy)


def beta_lasso(X, f_xy, lmb):
    """
    Function for calculating the beta parameters for a linear regression model using lasso regression. 
    Here, the scikit-learn function for calculating the beta parameters for lasso regression is used.
    """

    return linear_model.Lasso(lmb, fit_intercept=False).fit(X, f_xy).coef_


def mse(f_xy_pred, f_xy):
    """
    Function for calculating the mean squared error between the true data 'f_xy' and the predicted data 'f_xy_pred'.
    """

    return np.sum((f_xy - f_xy_pred) ** 2) / len(f_xy)


def r2_score(f_xy_pred, f_xy):
    """
    Function for calculating the R2 score between the true data 'f_xy' and the predicted data 'f_xy_pred'.
    """
    
    return 1 - (np.sum((f_xy - f_xy_pred) ** 2)) / (
        np.sum((f_xy - np.sum(f_xy) / len(f_xy)) ** 2)
    )


def ols_regression(x, y, f_xy, degree, plot=False, plot_gray=False):
    """
    Function for performing ordinary least squares regression on the dataset {x, y} with a polynomial of order 'degree'.
    
    The dataset is split into a training set (80%) and a test set (20%), and the design matrices for these sets are generated.
    
    The regression model is trained on the design matrix of the training set, where each column has been scaled by 
    subtracting the mean of the values in that column. The true output 'f_xy' is also scaled by subtracting its mean value.

    Finally, the training set and the test set are used to predict the output and evaluate the error of the trained model.
    """
    np.random.seed(2)
    # Split the dataset into a training set and a test set.
    x_train, x_test, y_train, y_test, f_xy_train, f_xy_test = train_test_split(
        x, y, f_xy, test_size=0.2
    )
    f_xy_scaling = np.mean(f_xy_train)

    # Generate the design matrix for the training and the test set.
    X_train, X_train_scaling = design_matrix(x_train, y_train, degree)
    X_test, X_test_scaling = design_matrix(x_test, y_test, degree)

    X, tot_X_scaling = design_matrix(x, y, degree)

    # Optimize the model parameters beta by training the model on the training set.
    b_ols = beta_ols(X_train - X_train_scaling, f_xy_train - f_xy_scaling)

    # Predict the output values for the training set and the test set.
    # Note that the test set is also scaled using the mean values from the training set.
    f_xy_pred_train = np.matmul(X_train - X_train_scaling, b_ols) + f_xy_scaling

    f_xy_pred_test = np.matmul(X_test - X_train_scaling, b_ols) + f_xy_scaling

    print("Beta coefficients: ")
    col = 0
    for i in range(1, degree + 1):
        for j in range(i + 1):
            print("beta(x^{} y^{}): {}".format(i - j, j, b_ols[col]))
            col += 1

    print(
        "MSE of training with polynomial degree %i samples with OLS: %.4f"
        % (degree, mse(f_xy_pred_train, f_xy_train))
    )

    print(
        "MSE of test with polynomial degree %i samples with OLS: %.4f"
        % (degree, mse(f_xy_pred_test, f_xy_test))
    )

    if plot:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Plot the surface of the fitted polynomial of order 'degree', and the true values as a scatter-plot.
        mesh_len = int(np.sqrt(len(x)))
        surf = ax.plot_surface(
            np.reshape(x, (mesh_len, mesh_len)),
            np.reshape(y, (mesh_len, mesh_len)),
            np.reshape(
                np.matmul(X - X_train_scaling, b_ols) + f_xy_scaling,
                (mesh_len, mesh_len),
            ),
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )
        ax.scatter(X_test[:, 0], X_test[:, 1], f_xy_test)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5, ax=ax)

        plt.tight_layout()
        plt.show()
        plt.close()

    if plot_gray:
        fig, ax = plt.subplots()
        f_xy_pred_tot = np.reshape(
            np.matmul(X - X_train_scaling, b_ols) + f_xy_scaling,
            (int(np.sqrt(len(X))), int(np.sqrt(len(X)))),
        )
        ax.imshow(f_xy_pred_tot, cmap="gray")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.tight_layout()
        plt.show()
        plt.close()

    return (
        b_ols,
        mse(f_xy_pred_train, f_xy_train),
        mse(f_xy_pred_test, f_xy_test),
        r2_score(f_xy_pred_train, f_xy_train),
        r2_score(f_xy_pred_test, f_xy_test),
    )


def ridge_regression(x, y, f_xy, degree, lmbda):
    """
    Similar function to 'ols_regression', performing training and test sets, training a model, 
    and finally predicting the output and evaluating the trained model. 
    
    While 'ols_regression' used ordinary least squares regression, this function uses ridge regression. 
    The function must therefore also be provided with a lambda value.
    """
    np.random.seed(2)
    x_train, x_test, y_train, y_test, f_xy_train, f_xy_test = train_test_split(
        x, y, f_xy, test_size=0.2
    )
    f_xy_scaling = np.mean(f_xy_train)

    X_train, X_train_scaling = design_matrix(x_train, y_train, degree)
    X_test, X_test_scaling = design_matrix(x_test, y_test, degree)

    b_ridge = beta_ridge(X_train - X_train_scaling, f_xy_train - f_xy_scaling, lmbda)

    f_xy_pred_train = np.matmul(X_train - X_train_scaling, b_ridge) + f_xy_scaling

    f_xy_pred_test = np.matmul(X_test - X_train_scaling, b_ridge) + f_xy_scaling

    print(
        "MSE of training samples with polynomial degree %i with Ridge (lambda = %.5f): %.5f"
        % (degree, lmbda, mse(f_xy_pred_train, f_xy_train))
    )

    print(
        "MSE of test samples with polynomial degree %i with Ridge (lambda = %.5f): %.5f"
        % (degree, lmbda, mse(f_xy_pred_test, f_xy_test))
    )

    return (
        b_ridge,
        mse(f_xy_pred_train, f_xy_train),
        mse(f_xy_pred_test, f_xy_test),
        r2_score(f_xy_pred_train, f_xy_train),
        r2_score(f_xy_pred_test, f_xy_test),
    )


def lasso_regression(x, y, f_xy, degree, lmbda):
    """
    Similar function to 'ols_regression', performing training and test sets, training a model, 
    and finally predicting the output and evaluating the trained model. 
    
    While 'ols_regression' used ordinary least squares regression, this function uses lasso regression. 
    The function must therefore also be provided with a lambda value.
    """
    np.random.seed(2)
    x_train, x_test, y_train, y_test, f_xy_train, f_xy_test = train_test_split(
        x, y, f_xy, test_size=0.2
    )
    f_xy_scaling = np.mean(f_xy_train)

    X_train, X_train_scaling = design_matrix(x_train, y_train, degree)
    X_test, X_test_scaling = design_matrix(x_test, y_test, degree)

    b_lasso = beta_lasso(X_train - X_train_scaling, f_xy_train - f_xy_scaling, lmbda)

    f_xy_pred_train = np.matmul(X_train - X_train_scaling, b_lasso) + f_xy_scaling

    f_xy_pred_test = np.matmul(X_test - X_train_scaling, b_lasso) + f_xy_scaling

    print(
        "MSE of training samples with polynomial degree %i with Lasso (lambda = %.5f): %.5f"
        % (degree, lmbda, mse(f_xy_pred_train, f_xy_train))
    )

    print(
        "MSE of test samples with polynomial degree %i with Lasso (lambda = %.5f): %.5f"
        % (degree, lmbda, mse(f_xy_pred_test, f_xy_test))
    )

    return (
        b_lasso,
        mse(f_xy_pred_train, f_xy_train),
        mse(f_xy_pred_test, f_xy_test),
        r2_score(f_xy_pred_train, f_xy_train),
        r2_score(f_xy_pred_test, f_xy_test),
    )


def bias_variance_tradeoff_bootstrap(x, y, f_xy, max_deg, num_bootstraps, plot=True):
    """
    Function for performing ordinary least squares regression with bootstrap resampling.

    The results from each bootstrap is used to calculate the bias and variance of the trained models.
    """
    np.random.seed(2023)
    x_train, x_test, y_train, y_test, f_xy_train, f_xy_test = train_test_split(
        x, y, f_xy, test_size=0.2
    )

    mse_list = []
    bias_list = []
    var_list = []

    for deg in range(1, max_deg + 1):
        # Perform the calculations for each polynomial degree between 1 and 'max_deg'.
        print("Deg: {}".format(deg))

        X_test, X_test_scaling = design_matrix(x_test, y_test, deg)

        f_xy_pred_test = np.zeros((len(f_xy_test), num_bootstraps))

        for i in range(num_bootstraps):
            # Resample the training set, train the model and predict the output of the test set. 
            # Performed 'num_bootstraps' number of times.
            x_boot, y_boot, f_xy_boot = resample(x_train, y_train, f_xy_train)

            X_boot, X_boot_scaling = design_matrix(x_boot, y_boot, deg)
            f_xy_scaling = np.mean(f_xy_boot)

            b_ols = beta_ols(X_boot - X_boot_scaling, f_xy_boot - f_xy_scaling)

            f_xy_pred_test[:, i] = (
                np.matmul(X_test - X_boot_scaling, b_ols) + f_xy_scaling
            )

        # Calculate the mean of the mean squared error for each resampled training set.
        mse_list.append(
            np.mean(
                [
                    np.mean((f_xy_pred_test[i, :] - f_xy_test[i]) ** 2)
                    for i in range(len(f_xy_test))
                ]
            )
        )
        # Calculate the mean of the bias of each data point in the test set.
        bias_list.append(
            np.mean(
                (
                    f_xy_test
                    - np.array(
                        [np.mean(f_xy_pred_test[i, :]) for i in range(len(f_xy_test))]
                    )
                )
                ** 2
            )
        )
        # Calculate the mean of the variance of each data point in the test set.
        var_list.append(
            np.mean([np.var(f_xy_pred_test[i, :]) for i in range(len(f_xy_test))])
        )

    if plot:
        # Plot the bias-variance tradeoff plot.
        fig, ax = plt.subplots(figsize=(16, 8))

        ax.plot(np.arange(1, max_deg + 1), mse_list, label="Mean squared error")
        ax.plot(np.arange(1, max_deg + 1), bias_list, label="Bias")
        ax.plot(np.arange(1, max_deg + 1), var_list, label="Variance")

        ax.set_xlabel("Degrees")
        ax.set_xticks(np.arange(1, max_deg + 1))
        ax.legend(fontsize=18, facecolor="white", framealpha=1)
        plt.tight_layout()
        plt.show()
        plt.close()

    return mse_list, bias_list, var_list


def k_fold(x, y, f_xy, k):
    """
    Function for splitting the dataset into 'k' close to equally sized subsets.
    These subsets will be used for k-fold cross-validation resampling, 
    where each subset has the role of test set exactly once, while the remaining subsets constitute the training set.
    """
    if k > len(x):
        print("The number of folds is too large.")
        sys.exit()

    num_samples = len(x)

    # Shuffling the dataset to distribute the data points in the subsets across the entire domain.
    rand_ids = np.arange(num_samples)
    np.random.shuffle(rand_ids)

    x_shuf = np.ravel(x[rand_ids])
    y_shuf = np.ravel(y[rand_ids])
    f_xy_shuf = np.ravel(f_xy[rand_ids])

    # Calculate the number of data points for each subset by using floor division, 
    # and then distributing the the remainder across the subsets
    fold_sizes = [num_samples // k] * k
    for i in range(num_samples % k):
        fold_sizes[i] += 1

    x_folds = []
    y_folds = []
    f_xy_folds = []

    start_index = 0
    for fold in range(k):
        # Each subset is created from the next 'fold_sizes[fold]' number of points in the dataset.
        # This is why it was important to shuffle the dataset before splitting it into subsets
        x_folds.append(list(x_shuf[start_index : start_index + fold_sizes[fold]]))
        y_folds.append(list(y_shuf[start_index : start_index + fold_sizes[fold]]))
        f_xy_folds.append(list(f_xy_shuf[start_index : start_index + fold_sizes[fold]]))

        start_index += fold_sizes[fold]

    return x_folds, y_folds, f_xy_folds


def training_with_crossval(x, y, f_xy, max_deg, k_folds, lambdas):
    """
    Function performing the k-fold cross-validation using ordinary least squares, ridge and lasso regression.
    """
    x_folds, y_folds, f_xy_folds = k_fold(x, y, f_xy, k_folds)
    mse_ols_list = []
    mse_ridge_list = [[] for n in range(len(lambdas))]
    mse_lasso_list = [[] for n in range(len(lambdas))]

    for deg in range(1, max_deg + 1):
        print("Deg: {}".format(deg))

        mse_ols_folds = []
        mse_ridge_folds = [[] for n in range(len(lambdas))]
        mse_lasso_folds = [[] for n in range(len(lambdas))]

        for i in range(k_folds):
            x_train = []
            y_train = []
            f_xy_train = []
            # Create the resampled training and test set.
            for j in range(k_folds):
                if j != i:
                    x_train += x_folds[j]
                    y_train += y_folds[j]
                    f_xy_train += f_xy_folds[j]
            
            # Train the model
            X_train, X_train_scaling = design_matrix(
                np.array(x_train), np.array(y_train), deg
            )

            f_xy_train_scaling = np.mean(f_xy_train)

            X_test, X_test_scaling = design_matrix(
                np.array(x_folds[i]), np.array(y_folds[i]), deg
            )

            f_xy_test = f_xy_folds[i]

            # Ordinary least squares
            b_ols = beta_ols(X_train - X_train_scaling, f_xy_train - f_xy_train_scaling)

            f_xy_pred_test = (
                np.matmul(X_test - X_train_scaling, b_ols) + f_xy_train_scaling
            )

            mse_ols_folds.append(np.mean((f_xy_pred_test - f_xy_test) ** 2))

            # Ridge
            for lmbda_id, lmbda in enumerate(lambdas):
                b_ridge = beta_ridge(
                    X_train - X_train_scaling, f_xy_train - f_xy_train_scaling, lmbda
                )

                f_xy_ridge_pred_test = (
                    np.matmul(X_test - X_train_scaling, b_ridge) + f_xy_train_scaling
                )

                mse_ridge_folds[lmbda_id].append(
                    np.mean((f_xy_ridge_pred_test - f_xy_test) ** 2)
                )

            # Lasso
            for lmbda_id, lmbda in enumerate(lambdas):
                b_lasso = beta_lasso(
                    X_train - X_train_scaling, f_xy_train - f_xy_train_scaling, lmbda
                )

                f_xy_lasso_pred_test = (
                    np.matmul(X_test - X_train_scaling, b_lasso) + f_xy_train_scaling
                )

                mse_lasso_folds[lmbda_id].append(
                    np.mean((f_xy_lasso_pred_test - f_xy_test) ** 2)
                )

        mse_ols_list.append(np.mean(mse_ols_folds))
        for lmbda_id in range(len(lambdas)):
            mse_ridge_list[lmbda_id].append(np.mean(mse_ridge_folds[lmbda_id]))
            mse_lasso_list[lmbda_id].append(np.mean(mse_lasso_folds[lmbda_id]))

    return mse_ols_list, mse_ridge_list, mse_lasso_list


def main():
    # Create the mesh grid for 'x' and 'y' using 21 equally spaced points between 0 and 1 for both 'x' and 'y'.
    x_mesh, y_mesh = np.meshgrid(np.linspace(0, 1, 21), np.linspace(0, 1, 21))
    x_mesh = np.ravel(x_mesh)
    y_mesh = np.ravel(y_mesh)

    # Calculate the Franke function across the entire mesh grid of data points.
    f_xy = franke_func(x_mesh, y_mesh, epsilon=0.1)

    max_poly_deg = 5

    max_num_betas = np.sum([deg + 1 for deg in range(1, max_poly_deg + 1)])
    beta_labels = []
    col = 0
    for i in range(1, max_poly_deg + 1):
        for j in range(i + 1):
            beta_labels.append(r"$\beta(x^{%i} y^{%i})$" % (i - j, j))
            col += 1

    # Part a) 
    # Plot the true output and the predicted output from ordinary least squares regression with polynomial degree = 5.
    beta, mse_train, mse_test, r2_train, r2_test = ols_regression(
        x_mesh, y_mesh, f_xy, 5, plot=True
    )

    # Plot the mean squared error and R2 score for OLS with polynomials of degrees between 1 and 5.
    betas = np.zeros((max_num_betas, max_poly_deg))
    mse_train = []
    mse_test = []
    r2_train = []
    r2_test = []

    for deg in range(1, max_poly_deg + 1):
        beta, mse_train_deg, mse_test_deg, r2_train_deg, r2_test_deg = ols_regression(
            x_mesh, y_mesh, f_xy, deg
        )

        for id, b in enumerate(beta):
            betas[id, deg - 1] = b
        mse_train.append(mse_train_deg)
        mse_test.append(mse_test_deg)
        r2_train.append(r2_train_deg)
        r2_test.append(r2_test_deg)

    fig, ax = plt.subplots(2, tight_layout=True)
    ax[0].plot(np.arange(1, len(mse_train) + 1), mse_train, label="Training samples")
    ax[0].plot(np.arange(1, len(mse_test) + 1), mse_test, label="Test samples")
    ax[0].set_xlabel("Polynomial degree")
    ax[0].set_ylabel("MSE")
    ax[0].set_xticks(np.arange(1, max_poly_deg + 1))
    ax[0].legend()

    ax[1].plot(np.arange(1, len(r2_train) + 1), r2_train, label="Training samples")
    ax[1].plot(np.arange(1, len(r2_test) + 1), r2_test, label="Test samples")
    ax[1].set_xlabel("Polynomial degree")
    ax[1].set_ylabel("R2 score")
    ax[1].set_xticks(np.arange(1, max_poly_deg + 1))
    ax[1].legend()

    plt.tight_layout()
    plt.show()
    plt.close()


    # Plot the model parameters beta for each polynomial degree.
    fig, ax = plt.subplots(tight_layout=True)
    for i in range(len(betas) - 6):
        deg = max_poly_deg - np.count_nonzero(betas[i]) + 1

        if i < ((len(betas) - 6) / 2):
            ax.plot(
                np.arange(deg, max_poly_deg + 1),
                [betas[i][j] for j in np.flatnonzero(betas[i])],
                label=beta_labels[i],
                linestyle="dotted",
            )
        # Latter half dashed lines
        if i >= ((len(betas) - 6) / 2):
            ax.plot(
                np.arange(deg, max_poly_deg + 1),
                [betas[i][j] for j in np.flatnonzero(betas[i])],
                label=beta_labels[i],
                linestyle="dashed",
            )

    ax.legend(ncol=3, fontsize=14, facecolor="white", framealpha=1)

    plt.tight_layout()
    plt.show()

    plt.close()

    # Part b)
    # Perform the same tasks as in "Part a)", except with ridge regression.
    lambda_vals = [0.00001, 0.0001, 0.001, 0.01]

    betas = np.zeros((max_num_betas, max_poly_deg))
    mse_train = np.zeros((len(lambda_vals), max_poly_deg))
    mse_test = np.zeros((len(lambda_vals), max_poly_deg))
    r2_train = np.zeros((len(lambda_vals), max_poly_deg))
    r2_test = np.zeros((len(lambda_vals), max_poly_deg))

    for deg in range(1, max_poly_deg + 1):
        for lmbda_id, lmbda in enumerate(lambda_vals):
            (
                beta,
                mse_train_deg,
                mse_test_deg,
                r2_train_deg,
                r2_test_deg,
            ) = ridge_regression(x_mesh, y_mesh, f_xy, deg, lmbda)

            for id, b in enumerate(beta):
                betas[id, deg - 1] = b
            mse_train[lmbda_id, deg - 1] = mse_train_deg
            mse_test[lmbda_id, deg - 1] = mse_test_deg
            r2_train[lmbda_id, deg - 1] = r2_train_deg
            r2_test[lmbda_id, deg - 1] = r2_test_deg

    fig1, ax1 = plt.subplots(1, tight_layout=True)
    fig2, ax2 = plt.subplots(1, tight_layout=True)
    for lmbda_id in range(len(lambda_vals)):
        ax1.plot(
            np.arange(1, max_poly_deg + 1),
            mse_train[lmbda_id],
            label="Training samples" + r"($\lambda = %.0e$)" % lambda_vals[lmbda_id],
        )
        ax1.plot(
            np.arange(1, max_poly_deg + 1),
            mse_test[lmbda_id],
            label="Test samples" + r"($\lambda = %.0e$)" % lambda_vals[lmbda_id],
        )

        ax2.plot(
            np.arange(1, max_poly_deg + 1),
            r2_train[lmbda_id],
            label="Training samples" + r"($\lambda = %.0e$)" % lambda_vals[lmbda_id],
        )
        ax2.plot(
            np.arange(1, max_poly_deg + 1),
            r2_test[lmbda_id],
            label="Test samples" + r"($\lambda = %.0e$)" % lambda_vals[lmbda_id],
        )

    ax1.set_xlabel("Polynomial degree")
    ax1.set_ylabel("MSE")
    ax1.set_xticks(np.arange(1, max_poly_deg + 1))
    # ax1.legend(bbox_to_anchor=(1, 1), fontsize=12)
    ax1.legend(fontsize=16)

    ax2.set_xlabel("Polynomial degree")
    ax2.set_ylabel("R2 score")
    ax2.set_xticks(np.arange(1, max_poly_deg + 1))
    # ax2.legend(bbox_to_anchor=(1, 1), fontsize=12)
    ax2.legend(fontsize=16)

    plt.tight_layout()
    plt.show()
    plt.close()

    fig, ax = plt.subplots(tight_layout=True)
    for i in range(len(betas) - 6):
        deg = max_poly_deg - np.count_nonzero(betas[i]) + 1

        if i < ((len(betas) - 6) / 2):
            ax.plot(
                np.arange(deg, max_poly_deg + 1),
                [betas[i][j] for j in np.flatnonzero(betas[i])],
                label=beta_labels[i],
                linestyle="dotted",
            )
        # Latter half dashed lines
        if i >= ((len(betas) - 6) / 2):
            ax.plot(
                np.arange(deg, max_poly_deg + 1),
                [betas[i][j] for j in np.flatnonzero(betas[i])],
                label=beta_labels[i],
                linestyle="dashed",
            )

    ax.legend(ncol=3, fontsize=14, facecolor="white", framealpha=1)

    plt.tight_layout()
    plt.show()
    plt.close()


    # Part c)
    # Perform the same tasks as in "Part a)", except with lasso regression.
    # Not plotting the beta values here.
    mse_train = np.zeros((len(lambda_vals), max_poly_deg))
    mse_test = np.zeros((len(lambda_vals), max_poly_deg))
    r2_train = np.zeros((len(lambda_vals), max_poly_deg))
    r2_test = np.zeros((len(lambda_vals), max_poly_deg))

    for deg in range(1, max_poly_deg + 1):
        for lmbda_id, lmbda in enumerate(lambda_vals):
            (
                beta,
                mse_train_deg,
                mse_test_deg,
                r2_train_deg,
                r2_test_deg,
            ) = lasso_regression(x_mesh, y_mesh, f_xy, deg, lmbda)

            mse_train[lmbda_id, deg - 1] = mse_train_deg
            mse_test[lmbda_id, deg - 1] = mse_test_deg
            r2_train[lmbda_id, deg - 1] = r2_train_deg
            r2_test[lmbda_id, deg - 1] = r2_test_deg

    fig1, ax1 = plt.subplots(1, tight_layout=True)
    fig2, ax2 = plt.subplots(1, tight_layout=True)
    for lmbda_id in range(len(lambda_vals)):
        ax1.plot(
            np.arange(1, max_poly_deg + 1),
            mse_train[lmbda_id],
            label="Training samples" + r"($\lambda = %.0e$)" % lambda_vals[lmbda_id],
        )
        ax1.plot(
            np.arange(1, max_poly_deg + 1),
            mse_test[lmbda_id],
            label="Test samples" + r"($\lambda = %.0e$)" % lambda_vals[lmbda_id],
        )

        ax2.plot(
            np.arange(1, max_poly_deg + 1),
            r2_train[lmbda_id],
            label="Training samples" + r"($\lambda = %.0e$)" % lambda_vals[lmbda_id],
        )
        ax2.plot(
            np.arange(1, max_poly_deg + 1),
            r2_test[lmbda_id],
            label="Test samples" + r"($\lambda = %.0e$)" % lambda_vals[lmbda_id],
        )

    ax1.set_xlabel("Polynomial degree")
    ax1.set_ylabel("MSE")
    ax1.legend(fontsize=16)

    ax2.set_xlabel("Polynomial degree")
    ax2.set_ylabel("R2 score")
    ax2.legend(fontsize=16)

    plt.show()
    plt.close()

    # Part e)
    # Perform the OLS regression for an increased number of polynomial degrees to se if we observe overfitting.
    max_poly_deg = 10
    max_num_betas = np.sum([deg + 1 for deg in range(1, max_poly_deg + 1)])

    betas = np.zeros((max_num_betas, max_poly_deg))
    mse_train = []
    mse_test = []
    r2_train = []
    r2_test = []

    for deg in range(1, max_poly_deg + 1):
        beta, mse_train_deg, mse_test_deg, r2_train_deg, r2_test_deg = ols_regression(
            x_mesh, y_mesh, f_xy, deg
        )

        for id, b in enumerate(beta):
            betas[id, deg - 1] = b

        mse_train.append(mse_train_deg)
        mse_test.append(mse_test_deg)

    fig, ax = plt.subplots(tight_layout=True)
    ax.plot(np.arange(1, len(mse_train) + 1), mse_train, label="Training samples")
    ax.plot(np.arange(1, len(mse_test) + 1), mse_test, label="Test samples")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("MSE")
    ax.set_xticks(np.arange(1, max_poly_deg + 1))
    ax.legend(facecolor="white", framealpha=1)

    plt.tight_layout()
    plt.show()
    plt.close()


    # Check the bias-variance tradeoff by performing 100 bootstrap resamplings.
    bias_variance_tradeoff_bootstrap(
        x_mesh, y_mesh, f_xy, max_deg=max_poly_deg, num_bootstraps=100
    )

    # Looking at how the bias-variance plot is changed by increasing the number of data points in the dataset 
    # (and thus in the mesh grid)
    x_mesh_more_points, y_mesh_more_points = np.meshgrid(
        np.linspace(0, 1, 101), np.linspace(0, 1, 101)
    )
    x_mesh_more_points = np.ravel(x_mesh_more_points)
    y_mesh_more_points = np.ravel(y_mesh_more_points)
    f_xy_more_points = franke_func(x_mesh_more_points, y_mesh_more_points, epsilon=0.1)

    # Decrease the number of bootstraps to 10.
    mse_boot, bias_boot, var_boot = bias_variance_tradeoff_bootstrap(
        x_mesh_more_points,
        y_mesh_more_points,
        f_xy_more_points,
        max_deg=max_poly_deg,
        num_bootstraps=10,
    )

    mse_boot, bias_boot, var_boot = bias_variance_tradeoff_bootstrap(
        x_mesh, y_mesh, f_xy, max_deg=max_poly_deg, num_bootstraps=10, plot=False
    )

    # Part f)

    k_folds = 10
    # Perform k-fold cross-validation with k = 10 folds, for OLS, ridge and lasso regression
    mse_ols, mse_ridge, mse_lasso = training_with_crossval(
        x_mesh, y_mesh, f_xy, max_poly_deg, k_folds, lambda_vals
    )

    # Plot mean squared error for prediction using k-fold cross-validation and using bootstrap resampling for OLS
    fig, ax = plt.subplots(tight_layout=True)

    ax.plot(
        np.arange(1, len(mse_ols) + 1),
        mse_ols,
        label="{}-fold cross validation".format(k_folds),
    )
    ax.plot(np.arange(1, len(mse_boot) + 1), mse_boot, label="{} bootstraps".format(10))
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("MSE")
    ax.set_xticks(np.arange(1, max_poly_deg + 1))
    ax.legend(facecolor="white", framealpha=1)

    plt.tight_layout()
    plt.show()
    plt.close()

    # Plot mean squared error of cross-validation for OLS, ridge and lasso regression
    fig, ax = plt.subplots(tight_layout=True)

    ax.plot(np.arange(1, len(mse_ols) + 1), mse_ols, label="OLS".format(k_folds))

    for id, val in enumerate(lambda_vals):
        ax.plot(
            np.arange(1, len(mse_ridge[id]) + 1),
            mse_ridge[id],
            label=r"Ridge ($\lambda$ = " + "{:.0e})".format(val),
        )

        ax.plot(
            np.arange(1, len(mse_lasso[id]) + 1),
            mse_lasso[id],
            label=r"Lasso ($\lambda$ = " + "{:.0e})".format(val),
        )

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("MSE")
    ax.set_xticks(np.arange(1, max_poly_deg + 1))
    ax.legend(fontsize=16, facecolor="white", framealpha=1)

    plt.tight_layout()
    plt.show()
    plt.close()

    # Part g)
    # Consider real topographic dataset.

    # Terrain close to Stavanger, Norway
    terrain_fpath = (
        os.path.dirname(os.path.realpath(__file__)) + "\\SRTM_data_Norway_2.tif"
    )
    terrain = imread(terrain_fpath)

    # Decrease the size of the dataset to decrease the computational cost of the calculations
    # Take the 1000x1000 first points in the image mesh, and then downsample this mesh by keeping every fifth point.
    N = 1000
    terrain = terrain[:N, :N][::5, ::5]

    # Plot the terrain data
    fig, ax = plt.subplots()
    ax.set_title('Terrain close to Stavanger, Norway')
    ax.imshow(terrain, cmap='gray')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.tight_layout()
    plt.show()
    plt.close()

    # Considering polynomial degrees up to 15.
    max_poly_deg = 15
    k_folds = 10

    x_mesh, y_mesh = np.meshgrid(
        np.linspace(0, 1, len(terrain[0])), np.linspace(0, 1, len(terrain[0]))
    )
    x_mesh = np.ravel(x_mesh)
    y_mesh = np.ravel(y_mesh)

    # Plot true data and the OLS regression fit with a polynomial of degree 100
    beta, mse_train, mse_test, r2_train, r2_test = ols_regression(
        x_mesh, y_mesh, np.ravel(terrain), 100, plot_gray=True
    )

    # 10-fold cross-validation of the terrain data, using OLS, ridge and lasso regression.
    mse_ols, mse_ridge, mse_lasso = training_with_crossval(
        x_mesh, y_mesh, np.ravel(terrain), max_poly_deg, k_folds, lambda_vals
    )

    # Plot the mean squared error of the cross-validation for OLS, ridge and lasso.
    fig, ax = plt.subplots(tight_layout=True)

    ax.plot(np.arange(1, len(mse_ols) + 1), mse_ols, label="OLS".format(k_folds))

    linestyles_lasso = ["solid", "dashed", "dotted"]
    for id, val in enumerate([0.00001, 0.0001, 0.001]):
        ax.plot(
            np.arange(1, len(mse_ridge[id]) + 1),
            mse_ridge[id],
            label=r"Ridge ($\lambda$ = " + "{:.0e})".format(val),
        )

        ax.plot(
            np.arange(1, len(mse_lasso[id]) + 1),
            mse_lasso[id],
            label=r"Lasso ($\lambda$ = " + "{:.0e})".format(val),
            linestyle=linestyles_lasso[id],
        )

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=16, facecolor="white", framealpha=1)
    plt.tight_layout()
    plt.show()

    plt.close()


if __name__ == "__main__":
    main()
