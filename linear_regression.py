from __future__ import print_function

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#======================================================================================================================#
# One variable
#======================================================================================================================#
def htheta(theta1e, x2e):
  return np.matmul(x2e, theta1e)


def fcost(theta1e, x2e, y1):
  h = htheta(theta1e, x2e)
  m = y1.size
  temp = h - y1
  return np.matmul(temp.transpose(), temp) / (2.0*m)


def fcost_part(theta1e, x2e, y1):
  h = htheta(theta1e, x2e)
  m = y1.size
  return np.matmul(x2e.transpose(), h - y1) / m


# Read matrix x2 and column vector y1, 1 means 1D array, 2 means 2D array.
# Here all the 1D arrays are considered to be column arrays.
data = np.genfromtxt('Data/data.csv', delimiter=',')
x2 = np.array([data[:, 0]]).transpose()
y1 = np.array([data[:, 1]]).transpose()

# x2 = np.array([[2104.0, 1416.0, 1534.0, 852.0]]).transpose()  # Here x2 is a 1D array, but for multiple variables, it is 2D
# y1 = np.array([[460.0, 232.0, 315.0, 178.0]]).transpose()
m = y1.size

x2e = np.hstack((np.ones((m, 1)), x2))  # extended x2 with 1 to represent x^0

print(x2e)

theta1e = np.array([[900, -0.1]]).transpose()
print(fcost(theta1e, x2e, y1))

##### Debug
hthetax = theta1e[0] + theta1e[1]*x2
t1 = hthetax - y1
print('theta1e', theta1e)
print('x2e', x2e)
print('t1', t1)
print('hthetax', hthetax)
print('htheta', htheta(theta1e, x2e))
print( 'J_explicit', np.sum( (hthetax-y1)**2 ) / (2*m) )
print('J', fcost(theta1e, x2e, y1))

print('J part explicit', np.sum(hthetax-y1)/m, np.sum( (hthetax-y1)*x2 )/m)
print( 'J part', fcost_part(theta1e, x2e, y1) )


# htheta = 

# cost_func = 

# Set initial theta1e
theta1e = np.array([[0, 0]]).transpose()
J_1 = fcost(theta1e, x2e, y1)
print(J_1)
alpha = 0.0001

# while True:
for i in range(2000):
  temp = theta1e - alpha * fcost_part(theta1e, x2e, y1)
  theta1e = temp
  J_1 = fcost(theta1e, x2e, y1)
  print(theta1e[0], theta1e[1], J_1)
#   if :
#     break

print(y1.transpose())

print(theta1e[0] + theta1e[1]*x2.transpose())
