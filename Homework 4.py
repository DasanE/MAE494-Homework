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


# Determine directions of feasible descent at the corner points of the feasible domain
# Show the gradient directions of f(x) and gi(s) at these points
# Verify graphical results analytically using the KKT conditions



## Problem 2
# Minimize x1, x2
# f(x1) = -x1

# Subject to:
# g1 -> x2 - (1-x1)^3 <= 0 AND x2 >= 0

# Graphically Sketch the Problem

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