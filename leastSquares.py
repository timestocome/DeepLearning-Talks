
# http://github.com/timestocome
# Linear Least Squares


import numpy as np
import matplotlib.pyplot as plt




##############################################################
# create test data
##############################################################

# data
features = np.asarray([1, 2, 3])
y = np.asarray([1, 2, 2])



##########################################################
# solve
#########################################################


# add row of ones to x, this is our bias
x_ones = np.ones(len(features))
X = np.vstack((x_ones, features))
XT = np.transpose(X)

w = np.ones(2)
learning_rate = 0.01

# print(X)
# print(XT)

XTX = np.dot(X, XT)
# print(XTX)



# goal
# J(w) = ||y - Xw||^2


# method
# dJ/dw = 2 * XTXw - 2 * XTy
# w = w - learning_rate * 2 * (XTXW - XTy)

error = 999
last_error = 9999

while (last_error - error) > 0.01:

    dw = learning_rate * 2 *  (np.dot(XTX, w) - np.dot(X,y))
    w = w - dw

    y_hat = np.dot(w, X)

    last_error = error
    error = np.sum((y - y_hat)**2)

    
    print(error, last_error)
