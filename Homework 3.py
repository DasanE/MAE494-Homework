## Homework 3 ##
import numpy as np
import torch as t
from torch.autograd import Variable

## Problem 1

# Constants
T = 20 #C
a_water = t.tensor([8.07131, 1730.63, 233.426])
a_dioxane = t.tensor([7.43155, 1554.679, 240.337])

psat_w = 10**(a_water[0]-(a_water[1])/(T+a_water[2]))
psat_d = 10**(a_dioxane[0]-(a_dioxane[1])/(T+a_dioxane[2]))
print("psat_w = ", psat_w)
print("psat_d = ", psat_d)

# Measured Data Table
x1 = t.tensor([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
x2 = t.tensor([1, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0])
p = t.tensor([28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.7, 17.5])

A_opt = Variable(t.tensor([1.0, 1.0]), requires_grad=True)

error = 10**(-3)
a = .0001

# Functions
def getPoint(A12, A21):
    return x1*t.exp(A12*(A21*x2/(A12*x1+A21*x2))**2)*psat_w + x2*t.exp(A21*(A12*x1/(A12*x1+A21*x2))**2)*psat_d

def loss_(p1):
    return (p1 - p)**2

# Since not linear, there is no analytical solution. Solve using gradient descent or Newton's method
for i in range(1000):
    pred = t.zeros(1,11)
    loss_pred = t.zeros(1,11)

    pred = getPoint(A_opt[0], A_opt[1])

    loss_pred = loss_(pred)

    loss = t.sum(loss_pred)
    print("Loss for current set is = ", loss)

    loss.backward()

    with t.no_grad():
        A_opt -= a * A_opt.grad
        A_opt.grad.zero_()

    print("New set of A values = ", A_opt)
    print(" ")

print("The final data set for A = ", A_opt.data.numpy())
print("The loss at this location = ", loss.data.numpy())
print("The gradiant at this location = ", A_opt.grad)

# Compare your optimized model with the data. Does your model fit well with the data
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

A12 = 1.958413
A21 = 1.6891907
psat_w = 17.47325
psat_d = 28.8241

fig = plt.figure()
ax = plt.axes(projection='3d')

def f(x,y):
    return x*np.exp(A12*(A21*y/(A12*x+A21*y))**2)*psat_w + y*np.exp(A21*(A12*x/(A12*x+A21*y))**2)*psat_d

x = np.linspace(0,2,100)
y = np.linspace(0,2,100)

X, Y = np.meshgrid(x,y)
Z = f(X, Y)

ax.contour3D(X,Y,Z,75)

xdata = t.tensor([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
ydata = t.tensor([1, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0])
zdata = t.tensor([28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.7, 17.5])

ax.scatter3D(xdata, ydata, zdata)

## Problem 2

#min x1,x2 = (4-2.1*x1^2+(x1^4)/3)*x1^2+x1*x2+(-4+4*x2^2)*x2^2

x1 = np.array([-3,3])
x2 = np.array([-2,2])
# Solve using Bayesian Optimization for













