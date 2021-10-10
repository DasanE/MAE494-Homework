## Homework 3 ##
import math
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

# Measured Data Table
x1 = Variable(t.tensor([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]), requires_grad=True)
x2 = Variable(t.tensor([1, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0]), requires_grad=True)
p = t.tensor([28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.7, 17.5])

#A = np.array([[0, 1],[1, 0]])
#A12 = A[1][2]
#A21 = A[2][1]
#print(str(A))
#A[1][2] = A12
#A[2][1] = A21
#print(str(A))
A12 = 1
A21 = 1
error = 10**(-3)

# Functions
def satPres(j):
    return x1[j]*math.exp(A12*((A12*x2[j])/(A12*x1[j]+A21*x2[j]))**2)*psat_w + x2[j]*math.exp(A21*((A12*x1[j])/(A12*x1[j]+A21*x2[j]))**2)*psat_d

def lossFun():
    i = 0
    loss_val = 0
    while i < np.size(x1, 1):
        x = (satPres(i)-p[i])**2
        loss_val = loss_val + x
        i += 1
    return loss_val

# Formulate the Least Square Problem (Estimate A12 and A21 using data from above table)

# Since not linear, there is no analytical solution. Solve using gradient descent or Newton's method
loss = lossFun()
a = .01
while loss.data.numpy > error:
    loss = lossFun()
    loss.backward()
    with t.no_grad():
            y = y - (a*x.grad)


# Compare optimized model with the data. Does your model fit well with the data






## Problem 2

#min x1,x2 = (4-2.1*x1^2+(x1^4)/3)*x1^2+x1*x2+(-4+4*x2^2)*x2^2

x1 = np.array([-3,3])
x2 = np.array([-2,2])
# Solve using Bayesian Optimization for













