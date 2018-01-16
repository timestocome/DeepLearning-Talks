
# http://github.com/timestocome
# Linear Regression


import numpy as np
import matplotlib.pyplot as plt




##############################################################
# create test data
##############################################################

# data
x = np.asarray([1, 2, 3,
                4, 5, 6])
y = np.asarray( 2 * x + 1)




##########################################################
# solve
#########################################################

# x vector transposed 
xT = np.transpose(x)


# equation 5.12
w = np.dot(xT, y) / np.dot(xT, x)


b = y[0] - np.dot(w, x)[0]


# 5.13
y_predicted = np.dot(w, x) + b



print('w ', w )
print('b', b)

#################################################
# plots
##################################################



plt.figure(figsize=(12,12))
plt.title('Linear Regression')

plt.scatter(x, y)
plt.plot(x, y_predicted)

plt.show()
