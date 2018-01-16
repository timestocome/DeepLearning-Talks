
# http://github.com/timestocome
# Linear Regression


# same as least squares except:
#   init weights using equation 5.12
#   add learning for b (intercept) separate from weights


import numpy as np
import matplotlib.pyplot as plt




##############################################################
# create test data
##############################################################

# data
x = np.asarray([1, 2, 3, 4, 5, 6])
y = np.asarray( 2 * x + 1)

# x vector transposed 
xT = np.transpose(x)



# init w Eq 5.12
w = np.dot(xT, y) / np.dot(xT, x)
b = 0.

##########################################################
# solve
#########################################################
error = 10.
learning_rate = 0.01


while(error > 0.01):

    w = w - learning_rate * (np.dot(xT, y) / np.dot(xT, x))

    y_predicted = np.dot(w, x) + b

    b = b - (y_predicted[0] - y[0]) 


    print('Error', error)
    error = np.sum( (y_predicted - y)**2 )

print('w (slope)', w )
print('b (intercept)', b)

#################################################
# plots
##################################################

y_predicted = x * w + b

plt.figure(figsize=(12,12))
plt.title('Linear Regression')

plt.scatter(x, y)
plt.plot(x, y_predicted)

plt.show()
