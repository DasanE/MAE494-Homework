## Homework 2 ##
import math
import os
import numpy as np

'''
THERE IS A PDF FILE ON GITHUB THAT HAS THE MISSING INFORMATION THAT COULDNT BE INSERTED INTO THE CODE 
'''

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

print('\n')

# Calculating the slopes of each set of points gets the following equations:
'''
s1 = 2x-1
s2 = .666x-.333
'''

# This means that in order reduce the function (f) you need to make sure that you head in the directions of either
'''
        s1 > 0 & s2 < 0
            or
        s1 < 0 & s2 > 0
        
        (ax1+bx2) < 0 & (cx1+dx2) > 0  or  (ax1+bx2) < 0 & (cx1+dx2) > 0
        where a = .6666  b = -1.5 and c = 2    d = -.5
'''

## Problem 2
# Find point on plane (x1 + 2x2 + 3x3 = 1) nearest to the point (-1, 0, 1)T , Is this a convex problem?
'''
Minimize (x1+1)^2 + x2^2 + (x3-1)^2
x1 = 1 - 2x2 - 3x3
n = <1,2,3>
Minimize (1-2x2-3x3+1)^2 + x2^2 + (x3-1)^2
'''

# Implement the gradient descent and Newton's algorithm for solving the problem.
# Attach your codes along with a short summary including (1) the initial points tested,
# (2) corresponding solutions, (3) a log-linear convergence plot.

# Gradient Descent
obj = lambda x2, x3: (1-2*x2-3*x3+1)**2 + x2**2 + (x3-1)**2
grad = lambda x2, x3: np.array([10*x2+12*x3-8, 12*x2+20*x3-14])
err = 1*10**(-3)
x0 = np.array([-1, -.125, .75])
k = 0
a = 1
soln = [x0]
x = soln[k]

g0 = grad(x[1],x[2])
error = math.sqrt((g0[0]**2+g0[1]**2))

def lineSearch(x2,x3,g):
    a = 1
    t = .8
    B = .5

    phi = lambda a, x2, x3: obj(x2,x3) - a*t*g.dot(g.T)

    def cfunc(a, x2, x3):
        c_array = np.array([x2,x3])-a*g
        return obj(c_array[0], c_array[1])

    while phi(a, x2, x3) < cfunc(a, x2, x3):
        a = a*B
    return a

while error >= err:

    slope = grad(x[1], x[2])
    a = lineSearch(x[1], x[2], slope)
    np.array(grad(x[1], x[2]))
    x = np.array([x[1], x[2]])-(a*np.array(grad(x[1], x[2])))
    x1 = np.array([1 - 2*x[0] - 3*x[1]])
    x = np.concatenate((x1, x), axis=None)
    soln.append(x)
    slope = grad(x[1], x[2])
    error = math.sqrt(slope[0]**2+slope[1]**2)
    k = k + 1

#print(str(soln))

print('\n')
print('The point closest to the plane is: ' + str(x))
print('This point has an error of: ' + str(error))
print('Algorithm ran for: ' + str(k) + ' iteration(s)')
print('\n')


# Newton's Algorithm (will converge since H is pd)
obj = lambda x2, x3: (1-2*x2-3*x3+1)**2 + x2**2 + (x3-1)**2
grad = lambda x2, x3: np.array([10*x2+12*x3-8, 12*x2+20*x3-14])
H = np.array([[10, 12],[12, 20]])
err = 1*10**(-3)
x0 = np.array([50, 25, 99])
k = 0
a = 1
soln = [x0]
x = soln[k]

g0 = grad(x[1],x[2])
error = math.sqrt((g0[0]**2+g0[1]**2))
H_inv = np.linalg.inv(H)

# function
def func_k(x1, x2, grad):

    return [x1, x2]-H_inv.dot(grad.T)

while error >= err:
    slope = grad(x[1], x[2])
    x = func_k(x[1], x[2], slope)
    x1 = np.array([1 - 2*x[0] - 3*x[1]])
    x = np.concatenate((x1, x), axis=None)
    np.array(grad(x[1], x[2]))
    soln.append(x)
    slope = grad(x[1], x[2])
    error = math.sqrt(slope[0]**2+slope[1]**2)
    k = k + 1

#print(str(soln))

print('\n')
print('The point closest to the plane is: ' + str(x))
print('This point has an error of: ' + str(error))
print('Algorithm ran for: ' + str(k) + ' iteration(s)')
print('\n')


## Problem 3
# f(x) and g(x) are two convex functions defined on the convex set X
# Prove that a*f(x) + b*g(x) is convex for a>0 and b>0
# In what conditions will F(g(x)) be convex?
'''
SEE PDF ON GITHUB FOR ANSWERS
'''

## Problem 4
# Show that f(x1) >= f(x0) + g(x1-x0)Txo for a convex function f(x): X -> R
'''
SEE PDF ON GITHUB FOR ANSWERS
'''

## Problem 5
# Consider an illumination problem: There are n lamps and m mirrors fixed to the ground. The target reflection intensity
# level is I(t). The actual reflection intensity level on the kth mirror can be computed as a^T_k*p where a_k is given
# by the distances between all lamps to the mirror, and p=[p_1,...,p_n]^T are the power output of the lamps. The objective
# is to keep the actual intensity levels as close to the target as possible by tuning the power output p.
# Formulate this problem as an optimization problem.
# Is your problem convex?
# If we require the overall power output of any of the $n$ lamps to be less than $p^*$, will the problem have a unique solution?
# If we require no more than half of the lamps to be switched on, will the problem have a unique solution?
'''
SEE PDF ON GITHUB FOR ANSWERS
'''