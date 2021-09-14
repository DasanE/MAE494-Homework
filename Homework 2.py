## Homework 2 ##
import math
import os

## Problem 1
# f = 2x1^2 - 4x1x2 + 1.5x2^2 + x2
# Show that the stationary point (Zero Gradiant) is a saddle (w/ Indefinite Hessian)
grad = [4*x[0]-4*x[1], -4*x[0]+3*x[1]+1]
hess = [[4, -4],
        [-4, 3]]

eig1 = (7+math.sqrt(7**2-4*1*0))/(2*1)
eig2 = (7-math.sqrt(7**2-4*1*0))/(2*1)

print(eig1  + ' ' + eig2 '/n')

# Find directions of downslopes away from the saddle (w/ Taylor's Expansion)


## Problem 2
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
