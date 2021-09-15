## Homework 2 ##
import math
import os
import numpy as np

## Problem 1
# f = 2x1^2 - 4x1x2 + 1.5x2^2 + x2
# Show that the stationary point (Zero Gradiant) is a saddle (w/ Indefinite Hessian)
x = np.array([0,0])
grad = [4*x[0]-4*x[1], -4*x[0]+3*x[1]+1]
hess = np.array([[4, -4],
                [-4, 3]])

sadl = np.array([1, 1])

eig1 = (7-math.sqrt(7**2-4*1*-4))/(2*1)
eig2 = (7+math.sqrt(7**2-4*1*-4))/(2*1)

print('\n')
print('First Eigenvalue = ' + str(eig1)  + ' Second Eigenvalue = ' + str(eig2))
print('Since the eigenvalues are both positive and negative this function is a saddle')
print('The saddle point is: ' + str(sadl))
print('\n')

# Find directions of downslopes away from the saddle (w/ Taylor's Expansion)
f0 = 2*sadl[0]**2 - 4*sadl[0]*sadl[1] + 1.5*sadl[1]**2 +sadl[1]
g0 = np.array([4*sadl[0]-4*sadl[1], -4*sadl[0]+3*sadl[1]+1])

# function
def f1(x0, x1):
        xa = np.array([x0, x1])
        '''
        fx = xa-sadl
        print('fx = ' + str(fx))
        print('hess = ' + str(hess))

        fa = 1/2*fx
        print('fa = ' + str(fa))

        fb = fa.dot(hess)
        print('fb = ' + str(fb))

        fc = fb.dot(fx.T)
        print('fc = ' + str(fc))
        '''
        return f0 + g0.dot(xa-sadl) +((1/2*(xa-sadl)).dot(hess)).dot((xa-sadl).T)

# Test out single point
'''
px1 = .625
px2 = .625
p1 = f1(px1, px2)
dp1 = p1-f0
print('Point (' + str(px1) + ',' + str(px2) + ') has a value of ' + str(p1) + ' which lies ' + str(dp1) + ' above/below the saddle point')
'''

n = 5
px1 = np.linspace(0,2,n)
px2 = np.linspace(0,2,n)
pth = np.zeros((n,n))
dpth = np.zeros((n,n))
i = 0
j = 0
while i < n:
    while j < n:
        pth[i][j] = f1(px1[i],px2[j])
        dpth[i][j] = pth[i][j]-f0
        print('Point (' + str(px1[i]) + ',' + str(px2[j]) + ') has a value of ' + str(pth[i][j]) + ' which lies ' + str(dpth[i][j]) + ' above/below the saddle point')
        j = j + 1
    i = i + 1
    j = 0

# taking these points and interpolating them to get where the function has zero change we get the following:
'''
(.5,0)
(1.5,2)
(.375,.286)
(1.625,1.714)
'''
# Calculating the slopes of each set of points gets the following equations:
'''
s1 = 2x-1
s2 = 1.1424x-.1424
'''

# This means that in order reduce the function (f) you need to make sure that you head in the directions of either
'''
        s1 > 0 & s2 < 0
            or
        s1 < 0 & s2 > 0
'''


# Problem 2
# Find point on plane (x1 + 2x2 + 3x3 = 1) nearest to the point (-1, 0, 1)T , Is this a convex problem?


# Implement the gradient descent and Newton's algorithm for solving the problem.
# Attach your codes along with a short summary including (1) the initial points tested,
# (2) corresponding solutions, (3) a log-linear convergence plot.



## Problem 3
# f(x) and g(x) are two convex functions defined on the convex set X
# Prove that a*f(x) + b*g(x) is convex for a>0 and b>0


# In what conditions will F(g(x)) be convex?


## Problem 4
# Show that f(x1) >= f(x0) + g(x1-x0)Txo for a convex function f(x): X -> R



## Problem 5
# Consider an illumination problem: There are n lamps and m mirrors fixed to the ground. The target reflection intensity
# level is I(t). The actual reflection intensity level on the kth mirror can be computed as a^T_k*p where a_k is given
# by the distances between all lamps to the mirror, and p=[p_1,...,p_n]^T are the power output of the lamps. The objective
# is to keep the actual intensity levels as close to the target as possible by tuning the power output p.

# Formulate this problem as an optimization problem.



# Is your problem convex?


# If we require the overall power output of any of the $n$ lamps to be less than $p^*$, will the problem have a unique solution?



# If we require no more than half of the lamps to be switched on, will the problem have a unique solution?
