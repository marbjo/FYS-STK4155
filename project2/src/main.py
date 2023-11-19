# author: Markus Bjørklund and Magnus Kristoffersen
# creation date: 2023-11-06
# encoding: utf-8
""" Main """


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import r2_score

from matplotlib.colors import LogNorm, Normalize

# import autograd.numpy as np
# from autograd import grad

# Own modules
import gradient_methods as gdm
import neural_net as nn
import loss_activation as lsa


"""Defining function to approximate and design matrix"""


def f_x(x):
    val = x**2 - 4 * x + 3
    return val


def mse(target, pred):
    t = np.ravel(target)
    p = np.ravel(pred)
    return np.sum((t - p) ** 2) / len(t)


noise_coeff = 0.1
n = 800  # Number of samples in training set
n_test = 200  # Number of samples in test set
start = 0
stop = 2
x = np.random.uniform(low=start, high=stop, size=(n, 1))  # Random x-values
x_test = np.random.uniform(low=start, high=stop, size=(n_test, 1))  # Random x-values

y_reg = f_x(x) + noise_coeff * np.random.randn(
    n, 1
)  # Function values for x^2 - 4x + 3, plus noise

y_reg_test = f_x(x_test) + noise_coeff * np.random.randn(
    n_test, 1
)  # Function values for x^2 - 4x + 3, plus noise

p = 3  # Polynomial degree 2 (3 predictors)

X = np.zeros((len(x), p))
X_test = np.zeros((len(x_test), p))

for i in range(p):
    X[:, i] = x[:, 0] ** (i)
    X_test[:, i] = x_test[:, 0] ** (i)


"""part a)"""

# Global ridge parameter
lmbda = 0.001

# Anayltic beta from expression
beta_an = np.array([[3], [-4], [1]])

##################
# ERROR VS LEARNING RATE
##################


tot_err_list_GD = []
tot_err_list_GD_mom = []
tot_err_list_SGD = []
tot_err_list_SGD_mom = []
tot_err_list_ADA = []
tot_err_list_ADA_mom = []
tot_err_list_ADA_SGD = []
tot_err_list_ADA_SGD_mom = []
tot_err_list_RMSProp_SGD = []
tot_err_list_ADAM_SGD = []

eta_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]

for eta_loop in eta_list:
    ###
    # PLAIN GRADIENT DESCENT
    ###

    beta_GD, counter_GD, beta_per_iter_GD = gdm.GD_plain(
        X, y_reg, n, p, type_="OLS", grad_type="an", lmb=0, eta=eta_loop, arg_seed=2023
    )

    diff_tot_GD = mse(y_reg_test, X_test @ beta_GD)
    tot_err_list_GD.append(diff_tot_GD)

    ###
    # GRADIENT DESCENT WITH MOMENTUM
    ###

    beta_GD_mom, counter_GD_mom, beta_per_iter_GD_mom = gdm.GD_momentum(
        X,
        y_reg,
        n,
        p,
        type_="OLS",
        grad_type="an",
        lmb=0,
        eta=eta_loop,
        gamma=0.5,
        arg_seed=2023,
    )

    diff_tot_GD_mom = mse(y_reg_test, X_test @ beta_GD_mom)
    tot_err_list_GD_mom.append(diff_tot_GD_mom)

    ###
    # STOCHASTIC GRADIENT DESCENT
    ###
    SGD_epoch = 5000

    beta_SGD, beta_per_iter_SGD = gdm.SGD(
        X,
        y_reg,
        n,
        p,
        type_="OLS",
        grad_type="an",
        epochs=SGD_epoch,
        M=5,
        eta=eta_loop,
        t0=200,
        t1=1500,
        lmb=0,
        arg_seed=2023,
    )

    diff_tot_SGD = mse(y_reg_test, X_test @ beta_SGD)
    tot_err_list_SGD.append(diff_tot_SGD)

    ###
    # STOCHASTIC GRADIENT DESCENT WITH MOMENTUM
    ###

    beta_SGD_mom, beta_per_iter_SGD_mom = gdm.SGD_momentum(
        X,
        y_reg,
        n,
        p,
        type_="OLS",
        grad_type="an",
        epochs=SGD_epoch,
        M=5,
        eta=eta_loop,
        t0=200,
        t1=1500,
        lmb=0,
        gamma=0.5,
        arg_seed=2023,
    )

    diff_tot_SGD_mom = mse(y_reg_test, X_test @ beta_SGD_mom)
    tot_err_list_SGD_mom.append(diff_tot_SGD_mom)

    ###
    # AdaGrad
    ###

    beta_ADA, counter_ADA, beta_per_iter_ADA = gdm.AdaGrad_GD(
        X,
        y_reg,
        n,
        p,
        type_="OLS",
        grad_type="an",
        lmb=0,
        eta=eta_loop,
        arg_seed=2023,
    )

    diff_tot_ADA = mse(y_reg_test, X_test @ beta_ADA)
    tot_err_list_ADA.append(diff_tot_ADA)

    ###
    # Adagrad with momentum
    ###
    beta_ADA_mom, counter_ADA_mom, beta_per_iter_ADA_mom = gdm.AdaGrad_GD_mom(
        X,
        y_reg,
        n,
        p,
        type_="OLS",
        grad_type="an",
        lmb=0,
        eta=eta_loop,
        gamma=0.5,
        arg_seed=2023,
    )

    diff_tot_ADA_mom = mse(y_reg_test, X_test @ beta_ADA_mom)
    tot_err_list_ADA_mom.append(diff_tot_ADA_mom)

    ###
    # AdaGrad with SGD
    ###
    SGD_epoch_ADA = SGD_epoch

    beta_ADA_SGD, beta_per_iter_ADA_SGD = gdm.AdaGrad_SGD(
        X,
        y_reg,
        n,
        p,
        type_="OLS",
        grad_type="an",
        epochs=SGD_epoch_ADA,
        M=5,
        lmb=0,
        eta=eta_loop,
        arg_seed=2023,
    )

    diff_tot_ADA_SGD = mse(y_reg_test, X_test @ beta_ADA_SGD)
    tot_err_list_ADA_SGD.append(diff_tot_ADA_SGD)

    ###
    # AdaGrad with SGD and momentum
    ###
    beta_ADA_SGD_mom, beta_per_iter_ADA_SGD_mom = gdm.AdaGrad_SGD_mom(
        X,
        y_reg,
        n,
        p,
        type_="OLS",
        grad_type="an",
        epochs=SGD_epoch_ADA,
        M=5,
        lmb=0,
        eta=eta_loop,
        gamma=0.5,
        arg_seed=2023,
    )

    diff_tot_ADA_SGD_mom = mse(y_reg_test, X_test @ beta_ADA_SGD_mom)
    tot_err_list_ADA_SGD_mom.append(diff_tot_ADA_SGD_mom)

    ###
    # RMSProp with SGD
    ###
    SGD_epoch_RMSProp = SGD_epoch

    beta_RMSProp_SGD, beta_per_iter_RMSProp_SGD = gdm.RMSProp_SGD(
        X,
        y_reg,
        n,
        p,
        type_="OLS",
        grad_type="an",
        epochs=SGD_epoch_RMSProp,
        M=5,
        lmb=0,
        eta=eta_loop,
        rho=0.99,
        arg_seed=2023,
    )

    diff_tot_RMSProp_SGD = mse(y_reg_test, X_test @ beta_RMSProp_SGD)
    tot_err_list_RMSProp_SGD.append(diff_tot_RMSProp_SGD)

    ###
    # ADAM with SGD
    ###
    SGD_epoch_ADAM = SGD_epoch
    beta_ADAM_SGD, beta_per_iter_ADAM_SGD = gdm.ADAM_SGD(
        X,
        y_reg,
        n,
        p,
        type_="OLS",
        grad_type="an",
        epochs=SGD_epoch_ADAM,
        M=5,
        lmb=0,
        eta=eta_loop,
        rho_1=0.9,
        rho_2=0.999,
        arg_seed=2023,
    )
    diff_tot_ADAM_SGD = mse(y_reg_test, X_test @ beta_ADAM_SGD)
    tot_err_list_ADAM_SGD.append(diff_tot_ADAM_SGD)

    print("Done with eta={}".format(eta_loop))

fig, ax = plt.subplots()


ax.plot(eta_list, tot_err_list_GD, label="GD")
ax.plot(eta_list, tot_err_list_GD_mom, label="GD momentum")
ax.plot(eta_list, tot_err_list_SGD, label="SGD")
ax.plot(eta_list, tot_err_list_SGD_mom, label="SGD momentum")
ax.plot(eta_list, tot_err_list_ADA, label="AdaGrad")
ax.plot(eta_list, tot_err_list_ADA_mom, label="AdaGrad momentum")
ax.plot(eta_list, tot_err_list_ADA_SGD, label="AdaGrad SGD")
ax.plot(eta_list, tot_err_list_ADA_SGD_mom, label="AdaGrad SGD momentum")
ax.plot(eta_list, tot_err_list_RMSProp_SGD, label="RMSProp SGD")
ax.plot(eta_list, tot_err_list_ADAM_SGD, label="ADAM SGD")
ax.legend(fontsize=10)
ax.grid()
ax.set_xlabel(r"$ \eta $", fontsize=24)
ax.set_ylabel("Mean Squared Error", fontsize=24)
ax.loglog()

plt.show()

#############
# RIDGE REGRESSION, LAMBDA VS LEARNING RATE
#############

SGD_epoch = 5000

eta_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
lmb_list = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

err_list = np.zeros((len(eta_list), len(lmb_list)))

counter2 = 0

for eta_loop in eta_list:
    counter = 0
    for lmb_loop in lmb_list:
        beta_SGD, beta_per_iter_SGD = gdm.SGD(
            X,
            y_reg,
            n,
            p,
            type_="ridge",
            grad_type="an",
            epochs=SGD_epoch,
            M=5,
            eta=eta_loop,
            lmb=lmb_loop,
            arg_seed=2023,
        )

        diff_tot_SGD = mse(y_reg_test, X_test @ beta_SGD)

        # Mask values above 1e10 to not ruin heatmap colorbar
        if diff_tot_SGD > 1e10:
            diff_tot_SGD = np.NaN

        err_list[counter2, counter] = diff_tot_SGD

        print(
            "element ({},{}) is lambda = {}, eta = {}".format(
                counter2, counter, lmb_loop, eta_loop
            )
        )
        counter += 1
    counter2 += 1

fig, ax = plt.subplots(figsize=(6.4 * 2, 4.8 * 2))
sns.heatmap(
    err_list,
    ax=ax,
    annot=True,
    fmt=".2e",
    xticklabels=lmb_list,
    yticklabels=eta_list,
    vmin=0,
    # norm=LogNorm(),
    annot_kws={"size": 16},
)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

for label in ax.xaxis.get_ticklabels():
    label.set_fontsize(20)

for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(20)

ax.set_ylabel(r"$\eta$", fontsize=24)
ax.set_xlabel(r"$\lambda$", fontsize=24)
plt.show()


#############
# TIME DECAY LEARNING RATE
#############

SGD_epoch = 5000

t0_list = [5, 10, 50, 100, 150, 200, 300]
t1_list = [200, 400, 600, 800, 1000, 1500]

t_err_list = np.zeros((len(t0_list), len(t1_list)))

counter2 = 0

for t0_ in t0_list:
    counter = 0
    for t1_ in t1_list:
        beta_SGD, beta_per_iter_SGD = gdm.SGD(
            X,
            y_reg,
            n,
            p,
            type_="OLS",
            grad_type="an",
            epochs=SGD_epoch,
            M=5,
            t0=t0_,
            t1=t1_,
            lmb=0,
            arg_seed=2023,
        )

        diff_tot_SGD = mse(y_reg_test, X_test @ beta_SGD)

        # Mask values above 1e10 to not ruin heatmap colorbar
        if diff_tot_SGD > 1e10:
            diff_tot_SGD = np.NaN

        t_err_list[counter2, counter] = diff_tot_SGD

        print(
            "element ({},{}) is t_1 = {}, t_0 = {}".format(counter2, counter, t1_, t0_)
        )
        counter += 1
    counter2 += 1

fig, ax = plt.subplots(figsize=(6.4 * 2, 4.8 * 2))
sns.heatmap(
    t_err_list,
    ax=ax,
    annot=True,
    fmt=".2e",
    xticklabels=t1_list,
    yticklabels=t0_list,
    vmin=0,
    norm=LogNorm(),
    annot_kws={"size": 16},
)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

for label in ax.xaxis.get_ticklabels():
    label.set_fontsize(20)

for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(20)

ax.set_ylabel(r"$t_0$", fontsize=24)
ax.set_xlabel(r"$t_1$", fontsize=24)
plt.show()


#############
# BATCH SIZE VS EPOCHS SGD
#############

batch_list = [5, 10, 20, 40, 100, 200]
epoch_list = [50, 125, 250, 500, 1000, 2000, 3000]

err_list = np.zeros((len(batch_list), len(epoch_list)))

counter2 = 0

for M_loop in batch_list:
    counter = 0
    for epoch_loop in epoch_list:
        beta_SGD, beta_per_iter_SGD = gdm.SGD(
            X,
            y_reg,
            n,
            p,
            type_="ridge",
            grad_type="an",
            epochs=epoch_loop,
            M=M_loop,
            eta=1e-3,
            lmb=0,
            arg_seed=2023,
        )

        diff_tot_SGD = mse(y_reg_test, X_test @ beta_SGD)

        # Mask values above 1e10 to not ruin heatmap colorbar
        if diff_tot_SGD > 1e10:
            diff_tot_SGD = np.NaN

        err_list[counter2, counter] = diff_tot_SGD

        print(
            "element ({},{}) is epoch = {}, M = {}".format(
                counter2, counter, epoch_loop, M_loop
            )
        )
        counter += 1
    counter2 += 1

fig, ax = plt.subplots(figsize=(6.4 * 2, 4.8 * 2))
sns.heatmap(
    err_list,
    ax=ax,
    annot=True,
    fmt=".2e",
    xticklabels=epoch_list,
    yticklabels=batch_list,
    vmin=0,
    # norm=LogNorm(),
    annot_kws={"size": 16},
)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

for label in ax.xaxis.get_ticklabels():
    label.set_fontsize(20)

for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(20)

ax.set_ylabel("Size of batches", fontsize=24)
ax.set_xlabel("Number of epochs", fontsize=24)
plt.show()


################
# ERROR VS ITERATION
################


###
# PLAIN GRADIENT DESCENT
###

beta_GD, counter_GD, beta_per_iter_GD = gdm.GD_plain(
    X, y_reg, n, p, type_="OLS", grad_type="an", lmb=0, eta=0.1, arg_seed=2023
)

diff_tot_GD = mse(y_reg_test, X_test @ beta_GD)
print("Final MSE value for GD: {}".format(diff_tot_GD))
diff_per_iter_GD = np.zeros(counter_GD)
for i in range(counter_GD):
    diff_per_iter_GD[i] = mse(y_reg_test, X_test @ beta_per_iter_GD[i])

iter_vec_GD = np.linspace(0, counter_GD - 1, counter_GD)

###
# GRADIENT DESCENT WITH MOMENTUM
###

beta_GD_mom, counter_GD_mom, beta_per_iter_GD_mom = gdm.GD_momentum(
    X,
    y_reg,
    n,
    p,
    type_="OLS",
    grad_type="an",
    lmb=0,
    eta=0.1,
    gamma=0.5,
    arg_seed=2023,
)

diff_tot_GD_mom = mse(y_reg_test, X_test @ beta_GD_mom)
print("Final MSE value for GD_mom: {}".format(diff_tot_GD_mom))
diff_per_iter_GD_mom = np.zeros(counter_GD_mom)
for i in range(counter_GD_mom):
    diff_per_iter_GD_mom[i] = mse(y_reg_test, X_test @ beta_per_iter_GD_mom[i])

iter_vec_GD_mom = np.linspace(0, counter_GD_mom - 1, counter_GD_mom)


###
# STOCHASTIC GRADIENT DESCENT
###
SGD_epoch = 5000

beta_SGD, beta_per_iter_SGD = gdm.SGD(
    X,
    y_reg,
    n,
    p,
    type_="OLS",
    grad_type="an",
    epochs=SGD_epoch,
    M=5,
    t0=50,
    t1=1500,
    lmb=0,
    arg_seed=2023,
)

diff_tot_SGD = mse(y_reg_test, X_test @ beta_SGD)
print("Final MSE value for SGD: {}".format(diff_tot_SGD))
diff_per_iter_SGD = np.zeros(SGD_epoch)
for i in range(SGD_epoch):
    diff_per_iter_SGD[i] = mse(y_reg_test, X_test @ beta_per_iter_SGD[i])

iter_vec_SGD = np.linspace(0, SGD_epoch - 1, SGD_epoch)

###
# STOCHASTIC GRADIENT DESCENT WITH MOMENTUM
###

beta_SGD_mom, beta_per_iter_SGD_mom = gdm.SGD_momentum(
    X,
    y_reg,
    n,
    p,
    type_="OLS",
    grad_type="an",
    epochs=SGD_epoch,
    M=5,
    t0=50,
    t1=1500,
    lmb=0,
    gamma=0.5,
    arg_seed=2023,
)


diff_tot_SGD_mom = mse(y_reg_test, X_test @ beta_SGD_mom)
print("Final MSE value for SGD_mom: {}".format(diff_tot_SGD_mom))
diff_per_iter_SGD_mom = np.zeros(SGD_epoch)
for i in range(SGD_epoch):
    diff_per_iter_SGD_mom[i] = mse(y_reg_test, X_test @ beta_per_iter_SGD_mom[i])

iter_vec_SGD_mom = np.linspace(0, SGD_epoch - 1, SGD_epoch)


###
# AdaGrad
###

beta_ADA, counter_ADA, beta_per_iter_ADA = gdm.AdaGrad_GD(
    X,
    y_reg,
    n,
    p,
    type_="OLS",
    grad_type="an",
    lmb=0,
    eta=0.1,
    arg_seed=2023,
)

diff_tot_ADA = mse(y_reg_test, X_test @ beta_ADA)
print("Final MSE value for AdaGrad: {}".format(diff_tot_ADA))
diff_per_iter_ADA = np.zeros(counter_ADA)
for i in range(counter_ADA):
    diff_per_iter_ADA[i] = mse(y_reg_test, X_test @ beta_per_iter_ADA[i])

iter_vec_ADA = np.linspace(0, counter_ADA - 1, counter_ADA)

###
# Adagrad with momentum
###
beta_ADA_mom, counter_ADA_mom, beta_per_iter_ADA_mom = gdm.AdaGrad_GD_mom(
    X,
    y_reg,
    n,
    p,
    type_="OLS",
    grad_type="an",
    lmb=0,
    eta=0.1,
    gamma=0.5,
    arg_seed=2023,
)

diff_tot_ADA_mom = mse(y_reg_test, X_test @ beta_ADA_mom)
print("Final MSE value for AdaGrad_mom: {}".format(diff_tot_ADA_mom))
diff_per_iter_ADA_mom = np.zeros(counter_ADA_mom)
for i in range(counter_ADA_mom):
    diff_per_iter_ADA_mom[i] = mse(y_reg_test, X_test @ beta_per_iter_ADA_mom[i])

iter_vec_ADA_mom = np.linspace(0, counter_ADA_mom - 1, counter_ADA_mom)

###
# AdaGrad with SGD
###
SGD_epoch_ADA = SGD_epoch

beta_ADA_SGD, beta_per_iter_ADA_SGD = gdm.AdaGrad_SGD(
    X,
    y_reg,
    n,
    p,
    type_="OLS",
    grad_type="an",
    epochs=SGD_epoch_ADA,
    M=5,
    lmb=0,
    eta=0.1,
    arg_seed=2023,
)

diff_tot_ADA_SGD = mse(y_reg_test, X_test @ beta_ADA_SGD)
print("Final MSE value for AdaGrad_SGD: {}".format(diff_tot_ADA_SGD))
diff_per_iter_ADA_SGD = np.zeros(SGD_epoch_ADA)
for i in range(SGD_epoch_ADA):
    diff_per_iter_ADA_SGD[i] = mse(y_reg_test, X_test @ beta_per_iter_ADA_SGD[i])

iter_vec_ADA_SGD = np.linspace(0, SGD_epoch_ADA - 1, SGD_epoch_ADA)

###
# AdaGrad with SGD and momentum
###
beta_ADA_SGD_mom, beta_per_iter_ADA_SGD_mom = gdm.AdaGrad_SGD_mom(
    X,
    y_reg,
    n,
    p,
    type_="OLS",
    grad_type="an",
    epochs=8000,
    M=5,
    lmb=0,
    eta=0.1,
    gamma=0.5,
    arg_seed=2023,
)

diff_tot_ADA_SGD_mom = mse(y_reg_test, X_test @ beta_ADA_SGD_mom)
print("Final MSE value for AdaGrad_SGD_mom: {}".format(diff_tot_ADA_SGD_mom))
diff_per_iter_ADA_SGD_mom = np.zeros(SGD_epoch_ADA)
for i in range(SGD_epoch_ADA):
    diff_per_iter_ADA_SGD_mom[i] = mse(
        y_reg_test, X_test @ beta_per_iter_ADA_SGD_mom[i]
    )

iter_vec_ADA_SGD_mom = np.linspace(0, SGD_epoch_ADA - 1, SGD_epoch_ADA)

###
# RMSProp with SGD
###
SGD_epoch_RMSProp = SGD_epoch

beta_RMSProp_SGD, beta_per_iter_RMSProp_SGD = gdm.RMSProp_SGD(
    X,
    y_reg,
    n,
    p,
    type_="OLS",
    grad_type="an",
    epochs=SGD_epoch_RMSProp,
    M=5,
    lmb=0,
    eta=1e-2,
    rho=0.99,
    arg_seed=2023,
)


diff_tot_RMSProp_SGD = mse(y_reg_test, X_test @ beta_RMSProp_SGD)
print("Final MSE value for RMSProp_SGD: {}".format(diff_tot_RMSProp_SGD))
diff_per_iter_RMSProp_SGD = np.zeros(SGD_epoch_RMSProp)
for i in range(SGD_epoch_RMSProp):
    diff_per_iter_RMSProp_SGD[i] = mse(
        y_reg_test, X_test @ beta_per_iter_RMSProp_SGD[i]
    )

iter_vec_RMSProp_SGD = np.linspace(0, SGD_epoch_RMSProp - 1, SGD_epoch_RMSProp)

###
# ADAM with SGD
###
SGD_epoch_ADAM = SGD_epoch
beta_ADAM_SGD, beta_per_iter_ADAM_SGD = gdm.ADAM_SGD(
    X,
    y_reg,
    n,
    p,
    type_="OLS",
    grad_type="an",
    epochs=SGD_epoch_ADAM,
    M=5,
    lmb=0,
    eta=1e-3,
    rho_1=0.9,
    rho_2=0.999,
    arg_seed=2023,
)

diff_tot_ADAM_SGD = mse(y_reg_test, X_test @ beta_ADAM_SGD)
print("Final MSE value for ADAM_SGD: {}".format(diff_tot_ADAM_SGD))
diff_per_iter_ADAM_SGD = np.zeros(SGD_epoch_ADAM)
for i in range(SGD_epoch_ADAM):
    diff_per_iter_ADAM_SGD[i] = mse(y_reg_test, X_test @ beta_per_iter_ADAM_SGD[i])

iter_vec_ADAM_SGD = np.linspace(0, SGD_epoch_ADAM - 1, SGD_epoch_ADAM)

###
# Plotting
###

fig, ax = plt.subplots()
ax.plot(iter_vec_GD, diff_per_iter_GD, label="GD")
ax.plot(iter_vec_GD_mom, diff_per_iter_GD_mom, label="GD momentum")
ax.plot(iter_vec_SGD, diff_per_iter_SGD, label="SGD")
ax.plot(iter_vec_SGD_mom, diff_per_iter_SGD_mom, label="SGD momentum")
ax.plot(iter_vec_ADA, diff_per_iter_ADA, label="AdaGrad")
ax.plot(iter_vec_ADA_mom, diff_per_iter_ADA_mom, label="AdaGrad momentum")  # Bad
ax.plot(iter_vec_ADA_SGD, diff_per_iter_ADA_SGD, label="AdaGrad SGD")
ax.plot(iter_vec_ADA_SGD_mom, diff_per_iter_ADA_SGD_mom, label="AdaGrad SGD momentum")
ax.plot(iter_vec_RMSProp_SGD, diff_per_iter_RMSProp_SGD, label="RMSProp SGD")
ax.plot(iter_vec_ADAM_SGD, diff_per_iter_ADAM_SGD, label="ADAM SGD")

ax.semilogx()
ax.set_ylabel("Mean Squared Error", fontsize=24)
ax.set_xlabel("Iterations / epochs", fontsize=24)
ax.grid()
ax.legend(fontsize=12)


fig, ax = plt.subplots()
ax.plot(iter_vec_GD, diff_per_iter_GD, label="GD")
ax.plot(iter_vec_GD_mom, diff_per_iter_GD_mom, label="GD momentum")
ax.plot(iter_vec_SGD, diff_per_iter_SGD, label="SGD")
ax.plot(iter_vec_SGD_mom, diff_per_iter_SGD_mom, label="SGD momentum")
ax.plot(iter_vec_ADA, diff_per_iter_ADA, label="AdaGrad")
ax.plot(iter_vec_ADA_mom, diff_per_iter_ADA_mom, label="AdaGrad momentum")  # Bad
ax.plot(iter_vec_ADA_SGD, diff_per_iter_ADA_SGD, label="AdaGrad SGD")
ax.plot(iter_vec_ADA_SGD_mom, diff_per_iter_ADA_SGD_mom, label="AdaGrad SGD momentum")
ax.plot(iter_vec_RMSProp_SGD, diff_per_iter_RMSProp_SGD, label="RMSProp SGD")
ax.plot(iter_vec_ADAM_SGD, diff_per_iter_ADAM_SGD, label="ADAM SGD")

ax.loglog()
ax.set_ylabel("Mean Squared Error", fontsize=24)
ax.set_xlabel("Iterations / epochs", fontsize=24)
ax.grid()
ax.legend(fontsize=12)

plt.show()

fig, ax1 = plt.subplots()
# ax1.plot(iter_vec_GD, diff_per_iter_GD, label="GD")
# ax1.plot(iter_vec_GD_mom, diff_per_iter_GD_mom, label="GD momentum")
ax1.plot(iter_vec_SGD, diff_per_iter_SGD, label="SGD")
ax1.plot(iter_vec_SGD_mom, diff_per_iter_SGD_mom, label="SGD momentum")
# ax1.plot(iter_vec_ADA, diff_per_iter_ADA, label="AdaGrad")
# ax1.plot(iter_vec_ADA_mom, diff_per_iter_ADA_mom, label="AdaGrad momentum")  # Bad
ax1.plot(iter_vec_ADA_SGD, diff_per_iter_ADA_SGD, label="AdaGrad SGD")
ax1.plot(iter_vec_ADA_SGD_mom, diff_per_iter_ADA_SGD_mom, label="AdaGrad SGD momentum")
ax1.plot(iter_vec_RMSProp_SGD, diff_per_iter_RMSProp_SGD, label="RMSProp SGD")
ax1.plot(iter_vec_ADAM_SGD, diff_per_iter_ADAM_SGD, label="ADAM SGD")

ax1.semilogy()
ax1.set_ylabel("Mean Squared Error", fontsize=24)
ax1.set_xlabel("Iterations / epochs", fontsize=24)
ax1.grid()
ax1.legend(fontsize=18)
ax1.set_xlim(left=0, right=2000)
plt.show()


# Predictions on uniformly distributed test set
x_predict = np.linspace(start, stop, 100)
X_predict = np.zeros((len(x_predict), p))
for i in range(p):
    X_predict[:, i] = x_predict ** (i)

analytic = X_predict @ beta_an
gd_predict_OLS = X_predict @ beta_GD


# Plotting
fig2, ax1 = plt.subplots(figsize=(6.4 * 2, 4.8 * 4))
ax1.scatter(x, y_reg, color="green", label="Data", alpha=0.4)
ax1.plot(
    x_predict,
    analytic,
    color="blue",
    linestyle="dashed",
    label="Analytic function\nwithout noise",
)
ax1.plot(
    x_predict, gd_predict_OLS, color="red", linestyle="dotted", label="Gradient descent"
)
ax1.legend()
ax1.set_title("Ordinary Least Squares", fontsize=24)
plt.show()


"""part b)"""

p = 2  # Polynomial degree 2. Bias is already baked into the neural net, no need for column in design matrix.

X = np.zeros((len(x), p))
for i in range(p):
    X[:, i] = x[:, 0] ** (i + 1)

# Reshaping data to fit with the neural net structure
y_reg = np.reshape(y_reg, (len(y_reg), 1))

# Testing data
test_x = np.linspace(start, stop, n)  # Test vector
test_X = np.zeros((len(test_x), p))  # Test design matrix

for i in range(p):
    test_X[:, i] = test_x ** (i + 1)


# Creating networks

# eta = 0.1  # Learning rate
momentum = 0  # Momentum parameter

input_n = p  # Number of nodes in input layer
hid_n = 3  # Number of nodes in hidder layer (2 or max 3)
output_n = 1  # Number of nodes in the output layer


# Loop over different values for eta and lambda
eta_list = [1e-4, 1e-3, 1e-2, 1e-1, 5e-1]
lmb_list = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
mse_arr = np.zeros((len(eta_list), len(lmb_list)))
r2_arr = np.zeros((len(eta_list), len(lmb_list)))
counter2 = 0


for eta_loop in eta_list:
    counter = 0
    for lmb_loop in lmb_list:
        net_reg = [
            nn.Layer(
                input_n,
                hid_n,
                nn.Fixed(eta_loop, momentum),
                nn.Fixed(eta_loop, momentum),
                arg_seed=0,
            ),
            nn.Activation_Layer(lsa.sigmoid, lsa.sigmoid_deriv),
            nn.Layer(
                hid_n,
                output_n,
                nn.Fixed(eta_loop, momentum),
                nn.Fixed(eta_loop, momentum),
                arg_seed=1,
            ),
            nn.Activation_Layer(lsa.linear, lsa.linear_deriv),
        ]

        # Train the network
        nn.train(
            net_reg,
            X,
            y_reg,
            lsa.mse_loss_deriv,
            lsa.l2_regularization_deriv,
            batches=1,
            epoch_n=10000,
            lambda_val=lmb_loop,
        )

        # Predict values based on testing data
        predictions = nn.predict(net_reg, test_X, f_x(test_x))

        # .predict returns a list of (1,1) numpy array elements...
        predictions = np.ravel(predictions)

        # MSE loss and R2-score
        mse_reg_own = lsa.mse_loss(predictions, f_x(test_x))
        r2_reg_own = r2_score(f_x(test_x), predictions)

        mse_arr[counter2, counter] = mse_reg_own
        r2_arr[counter2, counter] = r2_reg_own

        print("Done with eta={}, lmb={}".format(eta_loop, lmb_loop))

        counter += 1
    counter2 += 1

# Heatmap for MSE
fig, ax = plt.subplots(figsize=(6.4 * 2, 4.8 * 2))
sns.heatmap(
    mse_arr,
    ax=ax,
    annot=True,
    fmt=".2e",
    xticklabels=lmb_list,
    yticklabels=eta_list,
    vmin=0,
    # norm=LogNorm(),
    annot_kws={"size": 16},
)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

for label in ax.xaxis.get_ticklabels():
    label.set_fontsize(20)

for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(20)

ax.set_ylabel(r"$\eta$", fontsize=24)
ax.set_xlabel(r"$\lambda$", fontsize=24)

# Heatmap for R2
fig, ax = plt.subplots(figsize=(6.4 * 2, 4.8 * 2))
sns.heatmap(
    r2_arr,
    ax=ax,
    annot=True,
    fmt=".2e",
    xticklabels=lmb_list,
    yticklabels=eta_list,
    vmin=0,
    # norm=LogNorm(),
    annot_kws={"size": 16},
)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

for label in ax.xaxis.get_ticklabels():
    label.set_fontsize(20)

for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(20)

ax.set_ylabel(r"$\eta$", fontsize=24)
ax.set_xlabel(r"$\lambda$", fontsize=24)
plt.show()


""" # Scikit-learn neural net regressor

regr = MLPRegressor(
    hidden_layer_sizes=(hid_n,),
    activation="logistic",
    random_state=1,
    max_iter=100000,
    alpha=0.1,  # l2 regularization
    solver="sgd",
    learning_rate_init=0.1,  # Bad results from eta = 0.001 and 0.01
).fit(X, y_reg.ravel())
predictions_reg_scikit = regr.predict(test_X)

# MSE loss and R2-score
mse_reg_scikit = lsa.mse_loss(predictions_reg_scikit, f_x(test_x))
r2_reg_scikit = r2_score(f_x(test_x), predictions_reg_scikit)


# Plot
fig, ax = plt.subplots(figsize=(6.4 * 3, 4.8 * 3))
ax.plot(test_x, predictions, label="NN Predictions")  # Predicted by net
ax.plot(test_x, f_x(test_x), label="Analytic")  # Real function values
ax.plot(test_x, predictions_reg_scikit, label="Scikit-learn NN")  # Scikit-learn NN
ax.legend()
ax.grid()
plt.show() """


"""part c)"""
momentum = 0  # Momentum parameter

input_n = p  # Number of nodes in input layer
hid_n = 3  # Number of nodes in hidder layer (2 or max 3)
output_n = 1  # Number of nodes in the output layer

eta_list = [1e-4, 1e-3, 1e-2, 1e-1, 5e-1]
activation_labels = ["Sigmoid", "ReLU", "Leaky ReLU"]
alpha = 0.01
accuracy_one_layer_activations = []

for label_id, activation in enumerate(
    [
        (lsa.sigmoid, lsa.sigmoid_deriv),
        (lsa.RELU, lsa.RELU_deriv),
        (lsa.leaky_RELU(alpha), lsa.leaky_RELU_deriv(alpha)),
    ]
):
    counter = 0
    accuracy_one_layer = np.zeros(len(eta_list))
    print("Activation: {}".format(activation_labels[label_id]))
    for eta in eta_list:
        print("Eta = {}".format(eta))
        net_reg = [
            nn.Layer(input_n, hid_n, nn.Fixed(eta), nn.Fixed(eta), arg_seed=0),
            nn.Activation_Layer(activation[0], activation[1]),
            nn.Layer(hid_n, output_n, nn.Fixed(eta), nn.Fixed(eta), arg_seed=1),
            nn.Activation_Layer(lsa.linear, lsa.linear_deriv),
        ]

        # Train the network
        nn.train(
            net_reg,
            X,
            y_reg,
            lsa.mse_loss_deriv,
            lsa.l2_regularization_deriv,
            batches=10,
            epoch_n=10000,
            lambda_val=0,
        )

        # Predict values based on testing data
        predictions = nn.predict(net_reg, test_X, f_x(test_x))

        # .predict returns a list of (1,1) numpy array elements...
        predictions = np.ravel(predictions)

        # MSE loss and R2-score
        mse_reg_own = lsa.mse_loss(predictions, f_x(test_x))

        if mse_reg_own > 1e3:
            mse_reg_own = np.NaN
        # r2_reg_own = r2_score(f_x(test_x), predictions)

        accuracy_one_layer[counter] = mse_reg_own
        counter += 1
    accuracy_one_layer_activations.append(accuracy_one_layer)

fig, ax = plt.subplots(figsize=(6.4 * 2, 4.8 * 2))
sns.heatmap(
    np.array(accuracy_one_layer_activations),
    ax=ax,
    annot=True,
    fmt=".2e",
    xticklabels=eta_list,
    yticklabels=activation_labels,
    vmin=0,
    annot_kws={"size": 16},
)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

for label in ax.xaxis.get_ticklabels():
    label.set_fontsize(20)

for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(20)

ax.set_xlabel("Learning rate", fontsize=24)
ax.set_ylabel("Hidden layer activation", fontsize=24)
plt.show()
plt.close()


"""Part d)"""

######
# Wisconsin breast cancer data
######


# path of the given file
file_loc = Path(__file__).parent.absolute()
print(file_loc)

cancer_df = pd.read_csv(file_loc / "breast_cancer_data" / "data.csv")  # Load data set
# id_class = cancer_df[["id", "diagnosis"]]  # For later reference
cancer_class = cancer_df["diagnosis"]  # For later reference
cancer_class.replace(
    ("M", "B"), (1, 0), inplace=True
)  # Replace Malignant with 1, Benign with 0
class_data = cancer_class.to_numpy()

cancer_df = cancer_df.drop(
    ["id", "diagnosis", "Unnamed: 32"], axis=1
)  # Drop ID and classification, not real features (?). "Unnamed: 32" is just NaN

n_features = len(cancer_df.columns)
n_samples = len(cancer_df)

print("Number of features (columns): {}".format(n_features))
print("Number of samples (rows): {}".format(n_samples))

cancer_data = cancer_df.to_numpy()  # To numpy array

X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(
    cancer_data, class_data, test_size=0.2, random_state=0
)

# Scaling the input features
sc = StandardScaler()
X_train_cancer = sc.fit_transform(X_train_cancer)
X_test_cancer = sc.fit_transform(X_test_cancer)


# Reshaping data to fit neural net
X_cancer = np.reshape(
    X_train_cancer, (np.shape(X_train_cancer)[0], n_features)
)  # Input (design matrix)

Y_cancer = np.reshape(
    y_train_cancer, (np.shape(y_train_cancer)[0], 1)
)  # Targets for classification


X_cancer_test = np.reshape(
    X_test_cancer, (np.shape(X_test_cancer)[0], n_features)
)  # Input (design matrix)
Y_cancer_test = np.reshape(
    y_test_cancer, (np.shape(y_test_cancer)[0], 1)
)  # Targets for classification


eta = 0.001

input_n = n_features  # Number of nodes in input layer
output_n = 1  # Number of nodes in the output layer

################
# ACCURACY VS NODES AND ACTIVATION IN SINGLE HIDDEN LAYER
################

# 1 hidden layer with sigmoid, ReLU and leaky ReLU activation in the hidden layer
alpha = 0.1
accuracy_one_layer_activations = []
print("\nOne hidden layer")
activation_labels = ["Sigmoid", "ReLU", "Leaky ReLU"]

for label_id, activation in enumerate(
    [
        (lsa.sigmoid, lsa.sigmoid_deriv),
        (lsa.RELU, lsa.RELU_deriv),
        (lsa.leaky_RELU(alpha), lsa.leaky_RELU_deriv(alpha)),
    ]
):
    accuracy_one_layer = np.zeros(9)
    print("Activation: {}".format(activation_labels[label_id]))
    for id, nodes in enumerate(range(5, 50, 5)):
        print("Number of nodes: {}\n".format(nodes))
        hid_n = nodes  # Number of nodes in hidden layer
        net_cancer = [
            nn.Layer(input_n, hid_n, nn.Fixed(eta), nn.Fixed(eta), arg_seed=0),
            nn.Activation_Layer(activation[0], activation[1]),
            nn.Layer(hid_n, output_n, nn.Fixed(eta), nn.Fixed(eta), arg_seed=1),
            nn.Activation_Layer(lsa.sigmoid, lsa.sigmoid_deriv),
        ]

        # Train the network
        nn.train(
            net_cancer,
            X_cancer,
            Y_cancer,
            lsa.cross_entropy_loss_deriv,
            lsa.l2_regularization_deriv,
            batches=10,
            epoch_n=1000,
            lambda_val=0,
        )

        final_output = nn.predict(net_cancer, X_cancer_test, Y_cancer_test)
        final_output = np.ravel(final_output)

        # Round to closest 0 or 1
        final_output = np.round(final_output)
        final_output_target = np.ravel(Y_cancer_test)

        # Test accuracy score
        score = nn.accuracy(final_output, final_output_target)
        accuracy_one_layer[id] = score
    accuracy_one_layer_activations.append(accuracy_one_layer)

fig, ax = plt.subplots(figsize=(6.4 * 2, 4.8 * 2))
sns.heatmap(
    np.array(accuracy_one_layer_activations),
    ax=ax,
    annot=True,
    fmt=".4f",
    xticklabels=list(range(5, 50, 5)),
    yticklabels=activation_labels,
    annot_kws={"size": 20},
)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

for label in ax.xaxis.get_ticklabels():
    label.set_fontsize(20)

for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(20)

ax.set_xlabel("Nodes in layer", fontsize=24)
ax.set_ylabel("Hidden layer activation", fontsize=24)
plt.show()
plt.close()

################
# ACCURACY VS NODES AND ACTIVATIONS IN TWO HIDDEN LAYERS
################
# 2 hidden layers with sigmoid activation in the hidden layers
alpha = 0.1
accuracy_two_layers_activations = []
print("\nTwo hidden layers")
activation_labels = ["Sigmoid", "ReLU", "Leaky ReLU"]
for label_id, activation in enumerate(
    [
        (lsa.sigmoid, lsa.sigmoid_deriv),
        (lsa.RELU, lsa.RELU_deriv),
        (lsa.leaky_RELU(alpha), lsa.leaky_RELU_deriv(alpha)),
    ]
):
    accuracy_two_layers = np.zeros((3, 3))
    print("Activation: {}".format(activation_labels[label_id]))
    for id_1, nodes in enumerate(range(2, 12, 4)):
        for id_2, nodes2 in enumerate(range(2, 12, 4)):
            print("Number of nodes: ({}, {})\n".format(nodes, nodes2))
            hid_n = nodes  # Number of nodes in first hidden layer
            hid2_n = nodes2  # Number of nodes in second hidden layer
            net_cancer = [
                nn.Layer(input_n, hid_n, nn.Fixed(eta), nn.Fixed(eta), arg_seed=0),
                nn.Activation_Layer(activation[0], activation[1]),
                nn.Layer(hid_n, hid2_n, nn.Fixed(eta), nn.Fixed(eta), arg_seed=1),
                nn.Activation_Layer(activation[0], activation[1]),
                nn.Layer(hid2_n, output_n, nn.Fixed(eta), nn.Fixed(eta), arg_seed=2),
                nn.Activation_Layer(lsa.sigmoid, lsa.sigmoid_deriv),
            ]

            # Neural net
            nn.train(
                net_cancer,
                X_cancer,
                Y_cancer,
                lsa.cross_entropy_loss_deriv,
                lsa.l2_regularization_deriv,
                batches=10,
                epoch_n=1000,
                lambda_val=0,
            )

            final_output = nn.predict(net_cancer, X_cancer_test, Y_cancer_test)
            final_output = np.ravel(final_output)

            # Round to closest 0 or 1
            final_output = np.round(final_output)
            final_output_target = np.ravel(Y_cancer_test)

            # Test accuracy score
            score = nn.accuracy(final_output, final_output_target)
            accuracy_two_layers[id_1, id_2] = score

    accuracy_two_layers_activations.append(np.ravel(accuracy_two_layers))


fig, ax = plt.subplots(figsize=(6.4 * 2, 4.8 * 2))
sns.heatmap(
    np.array(accuracy_two_layers_activations),
    ax=ax,
    annot=True,
    fmt=".4f",
    xticklabels=[
        "(2,2)",
        "(2,6)",
        "(2,10)",
        "(6,2)",
        "(6,6)",
        "(6,10)",
        "(10,2)",
        "(10,6)",
        "(10,10)",
    ],
    yticklabels=activation_labels,
    annot_kws={"size": 20},
)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

for label in ax.xaxis.get_ticklabels():
    label.set_fontsize(20)

for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(20)

ax.set_xlabel("Nodes in hidden layers", fontsize=24)
ax.set_ylabel("Hidden layer activations", fontsize=24)
plt.show()
plt.close()

################
# LEARNING RATE VS REGULARIZATION
################

# 1 hidden layer with sigmoid, ReLU and leaky ReLU activation in the hidden layer

accuracy = np.zeros((7, 6))
print("\nOne hidden layer with 20 nodes, sigmoid activation, varying eta and lambda")
activation_labels = ["Sigmoid", "ReLU", "Leaky ReLU"]

for eta_id, eta in enumerate(np.logspace(-5, 1, 7)):
    print("Eta: {}".format(eta))
    for lambda_id, lambda_val in enumerate(np.logspace(-4, 1, 6)):
        print("Lambda: {}\n".format(lambda_val))
        hid_n = 20  # Number of nodes in hidder layer (Based on 2/3 of input features)
        net_cancer = [
            nn.Layer(input_n, hid_n, nn.Fixed(eta), nn.Fixed(eta), arg_seed=0),
            nn.Activation_Layer(lsa.sigmoid, lsa.sigmoid_deriv),
            nn.Layer(hid_n, output_n, nn.Fixed(eta), nn.Fixed(eta), arg_seed=1),
            nn.Activation_Layer(lsa.sigmoid, lsa.sigmoid_deriv),
        ]

        # Train the network
        nn.train(
            net_cancer,
            X_cancer,
            Y_cancer,
            lsa.cross_entropy_loss_deriv,
            lsa.l2_regularization_deriv,
            batches=10,
            epoch_n=1000,
            lambda_val=lambda_val,
        )

        final_output = nn.predict(net_cancer, X_cancer_test, Y_cancer_test)
        final_output = np.ravel(final_output)

        # Round to closest 0 or 1
        final_output = np.round(final_output)
        final_output_target = np.ravel(Y_cancer_test)

        # Test accuracy score
        score = nn.accuracy(final_output, final_output_target)
        accuracy[eta_id, lambda_id] = score

fig, ax = plt.subplots(figsize=(6.4 * 2, 4.8 * 2))
sns.heatmap(
    accuracy,
    ax=ax,
    annot=True,
    fmt=".4f",
    xticklabels=np.logspace(-4, 1, 6),
    yticklabels=np.logspace(-5, 1, 7),
    annot_kws={"size": 20},
)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

for label in ax.xaxis.get_ticklabels():
    label.set_fontsize(20)

for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(20)

ax.set_xlabel("Regularization", fontsize=24)
ax.set_ylabel("Learning rate", fontsize=24)
plt.show()
plt.close()

# Scikit-learn neural net

""" # Create neural net
clf = MLPClassifier(
    solver="sgd",
    alpha=0,
    hidden_layer_sizes=(20, 1),
    random_state=0,
    learning_rate_init=0.001,  # Default 0.001
    max_iter=1000,  # Default 200. 500 Still gives 10 decimal precision, but convergence warning
)

# Fit and make predictions on test data
clf.fit(X_cancer, Y_cancer.ravel())
predictions_scikit_nn = clf.predict(X_cancer_test)

# Test accuray score
score_scikit_nn = nn.accuracy(predictions_scikit_nn, np.ravel(Y_cancer_test))
print("Classification score for sci-kit neural net is {:.10f}".format(score_scikit_nn)) """

"""Part e)"""
# Logistic regression

logistic_accuracy = np.zeros((7, 6))
for eta_id, eta in enumerate(np.logspace(-5, 1, 7)):
    print("Eta: {}".format(eta))
    for lambda_id, lambda_val in enumerate(np.logspace(-4, 1, 6)):
        print("Lambda: {}\n".format(lambda_val))

        beta_logistic, beta_per_iter = gdm.SGD(
            X_cancer,
            Y_cancer,
            np.shape(X_cancer)[0],
            np.shape(X_cancer)[1],
            type_="log",
            grad_type="an",
            epochs=1000,
            M=5,
            eta=eta,
            lmb=lambda_val,
            arg_seed=0,
        )
        prediction_logistic = X_cancer_test @ beta_logistic
        rav = np.ravel(prediction_logistic)

        prediction_logistic = np.round(lsa.sigmoid(rav))
        score_logistic = nn.accuracy(prediction_logistic, np.ravel(Y_cancer_test))
        logistic_accuracy[eta_id, lambda_id] = score_logistic
        print(score_logistic)

fig, ax = plt.subplots(figsize=(6.4 * 2, 4.8 * 2))
sns.heatmap(
    logistic_accuracy,
    ax=ax,
    annot=True,
    fmt=".4f",
    xticklabels=np.logspace(-4, 1, 6),
    yticklabels=np.logspace(-5, 1, 7),
    annot_kws={"size": 20},
)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

for label in ax.xaxis.get_ticklabels():
    label.set_fontsize(20)

for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(20)

ax.set_xlabel(r"$\lambda$", fontsize=24)
ax.set_ylabel(r"$\eta$", fontsize=24)
plt.show()
plt.close()

# Scikit-learn logistic regression

"""
clf = LogisticRegression(random_state=0).fit(X_cancer, Y_cancer.ravel())
predictions_scikit = clf.predict(X_cancer_test)

score_scikit = nn.accuracy(predictions_scikit, np.ravel(Y_cancer_test))

print("Classification score from Scikit-learn's logistic regression {:.4f}".format(score_scikit))
"""
