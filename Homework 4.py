## Homework 4 ##


## Problem 1
# Minimize x1, x2
# f(x1, x2) = (x1+1)^2 + (x2-2)^2

# Subject to:
# g1 -> x1 - 2 <= 0
# g2 -> x2 - 1 <= 0
# g3 -> -x1 <= 0
# g4 -> -x2 <= 0

# Graphically Sketch the Problem
import torch as t
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

x1_down = 0
x1_up = 2
x2_down = 0
x2_up = 1
obj_space = .5

x1 = np.linspace(x1_down,x1_up,100)
x2 = np.linspace(x2_down,x2_up,100)
obj_x1 = np.linspace(x1_down-obj_space,x1_up+obj_space,200)
obj_x2 = np.linspace(x2_down-obj_space,x2_up+obj_space,200)

fig = plt.figure()
ax = plt.axes(projection = '3d')
plt.xlabel("X1")
plt.ylabel("X2")

def f(x1, x2):
    return (x1+1)**2 + (x2-2)**2


obj_X1, obj_X2 = np.meshgrid(obj_x1, obj_x2)
obj_Z = f(obj_X1, obj_X2)

ax.plot_surface(obj_X1, obj_X2, obj_Z, rstride = 1, cstride = 1, cmap = 'viridis', edgecolor = 'none')
ax.view_init(30,135)


#X1, X2 = np.meshgrid(x1, x2)
obj1_up = f(x1_up, x2)
obj1_down = f(x1_down, x2)
obj2_up = f(x1, x2_up)
obj2_down = f(x1, x2_down)

ax.plot(np.linspace(x1_up, x1_up, 100), x2, obj1_up, color = 'red')
ax.plot(np.linspace(x1_down, x1_down, 100), x2, obj1_down, color = 'blue')

ax.plot(x1, np.linspace(x2_up, x2_up, 100), obj2_up, color = 'red')
ax.plot(x1, np.linspace(x2_down, x2_down, 100), obj2_down, color = 'blue')

# Determine directions of feasible descent at the corner points of the feasible domain
# Show the gradient directions of f(x) and gi(s) at these points
# Verify graphical results analytically using the KKT conditions



## Problem 2
# Minimize x1, x2
# f(x1) = -x1

# Subject to:
# g1 -> x2 - (1-x1)^3 <= 0 AND x2 >= 0

# Graphically Sketch the Problem
import torch as t
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

array_size = 25

fig = plt.figure()
ax = plt.axes(projection = '3d')
plt.xlabel("X1")
plt.ylabel("X2")

x1 = np.linspace(-5,5,array_size)
x2 = np.linspace(0,5, array_size)

def f(x1):
    return -1*x1

def opt_cond(x1, x2):
    return x2 - (1-x1)**3

X1, X2 = np.meshgrid(x1,x2)
z = f(x1)
Z = np.repeat(z[:, np.newaxis], array_size, axis = 1)
Z = np.transpose(Z)
ax.plot_surface(X1, X2, Z, rstride = 1, cstride = 1, cmap = 'viridis', edgecolor = 'none')

opt_z = opt_cond(x1, x2)
opt_Z = np.repeat(opt_z[:, np.newaxis], array_size, axis = 1)
Z = np.transpose(opt_Z)
ax.plot_surface(X1, X2, opt_Z, rstride = 1, cstride = 1, cmap = 'viridis', edgecolor = 'none', alpha = .25)

ax.view_init(10,5)

# Find solution


## Problem 3
# Minimize x1, x2, x3
# f(x1, x2, x3) = x1x2 + x2x3 + x1x3

# Subject to:
# h -> x1 + x2 + x3 - 3 = 0

# Find the local solution (Use two methods: reduced gradient and lagrange multipliers)



## Problem 4
# Minimize x1, x2
# f(x) = 2x1 + bx2

# Subject to:
# g1 -> x1^2 + x2^2 -5 <= 0
# g2 -> x1 - x2 -2 <= 0

# Use reduced gradient to find the value(s) of the parameter b for which the point x_1=1, x_2=2 is the solution to the problem



## Problem 5
# Minimize x1, x2, x3
# f(x) = x1^2 + x2^2 + x3^2

# Subject to:
# h1 -> x1^2/4 + x2^2/5 + x3^2/25 - 1 = 0
# h2 -> x1 + x2 - x3 = 0

# Find the solutions by implementing the generalized reduced gradient algorithm