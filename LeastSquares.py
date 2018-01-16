
# http://github.com/timestocome
# Linear Least Squares

# Example taken from
# http://mlwiki.org/index.php/OLS_Regression

import numpy as np
import matplotlib.pyplot as plt




##############################################################
# create test data
##############################################################

# data
x = np.asarray([1, 2, 3, 4, 5, 6])
y = np.asarray(2 * x + 1)



##########################################################
# solve
#########################################################


# add row of ones to x, this is our bias
x_ones = np.ones(len(x))

# x inputs
X = np.vstack((x_ones, x))

# x vector transposed for dot product
XT = np.transpose(X)

# initial weight settings
w = np.ones(2)

# slows down weight adjustments to prevent overshooting mins
learning_rate = 0.01

# print(X)
# print(XT)

# X * X
XTX = np.dot(X, XT)
# print(XTX)



# goal
# J(w) = ||y - Xw||^2


# method
# dJ/dw = 2 * XTXw - 2 * XTy
# w = w - learning_rate * 2 * (XTXW - XTy)


# set to any high value or loop will stop on first pass
error = 999
last_error = 9999


# while error is decreasing
while (last_error - error) > 0.01:

    # use gradient to decide how much and which way to change weights
    dw = learning_rate * 2 *  (np.dot(XTX, w) - np.dot(X,y))
    w = w - dw

    # test predictions
    y_hat = np.dot(w, X)

    # update error 
    last_error = error
    error = np.sum((y - y_hat)**2)

    
    print(error)

y_predicted = np.dot(w, X)
print('w ', w )


#################################################
# plots
##################################################



plt.figure(figsize=(12,12))
plt.title('Least Squares' )

plt.scatter(x, y)
plt.plot(x, y_predicted)

plt.show()
