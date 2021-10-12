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

# Measured Data Table
x1 = Variable(t.tensor([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]))
x2 = Variable(t.tensor([1, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0]))
p = t.tensor([28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.7, 17.5])

A_opt = Variable(t.tensor([1.0, 1.0]), requires_grad=True)

error = 10**(-3)
a = .01

# Functions
def getPoint():
    i = 0
    val1 = 0
    while i < 11:
        val1[i] = x1[i]*math.exp(A12*(A12*x2[i]/(A12*x1[i]+A21*x2[i]))**2)*psat_w + x2[i]*math.exp(A21*((A12*x1[i])/(A12*x1[i]+A21*x2[i]))**2)*psat_d
        i = i + 1
    return val1

def loss(p1):
    return sum((p1 - p)**2)

# Formulate the Least Square Problem (Estimate A12 and A21 using data from above table)
def getPoint(A12, A21, i):
    return x1[i]*t.exp(A12*(A12*x2[i]/(A12*x1[i]+A21*x2[i]))**2)*psat_w + x2[i]*t.exp(A21*((A12*x1[i])/(A12*x1[i]+A21*x2[i]))**2)*psat_d

def loss(p1, i):
    return (p1 - p[i])**2

pred0 = getPoint(A_opt[0], A_opt[1],0)
pred1 = getPoint(A_opt[0], A_opt[1],1)
pred2 = getPoint(A_opt[0], A_opt[1],2)
pred3 = getPoint(A_opt[0], A_opt[1],3)
pred4 = getPoint(A_opt[0], A_opt[1],4)
pred5 = getPoint(A_opt[0], A_opt[1],5)
pred6 = getPoint(A_opt[0], A_opt[1],6)
pred7 = getPoint(A_opt[0], A_opt[1],7)
pred8 = getPoint(A_opt[0], A_opt[1],8)
pred9 = getPoint(A_opt[0], A_opt[1],9)
pred10 = getPoint(A_opt[0], A_opt[1],10)

loss0 = loss(pred0,0)
loss1 = loss(pred1,1)
loss2 = loss(pred2,2)
loss3 = loss(pred3,3)
loss4 = loss(pred4,4)
loss5 = loss(pred5,5)
loss6 = loss(pred6,6)
loss7 = loss(pred7,7)
loss8 = loss(pred8,8)
loss9 = loss(pred9,9)
loss10 = loss(pred10,10)

loss = loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9+loss10

loss.backward()

print(A_opt.grad)
# ([-980.2328, 197.636])

# Since not linear, there is no analytical solution. Solve using gradient descent or Newton's method




# Compare optimized model with the data. Does your model fit well with the data





## Problem 2

#min x1,x2 = (4-2.1*x1^2+(x1^4)/3)*x1^2+x1*x2+(-4+4*x2^2)*x2^2

x1 = np.array([-3,3])
x2 = np.array([-2,2])
# Solve using Bayesian Optimization for













