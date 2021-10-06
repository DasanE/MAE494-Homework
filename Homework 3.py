# Homework 3

## Problem 1

p = x1*exp(A12((A21*x2)/(A12*x1+A21*x2))^2)*psat_water + x2*exp(A21((A12*x1)/(A12*x1+A21*x2))^2)*psat_dioxane
log10(psat) = a1-a2/(T+a3)
T = 20 C

[a1,a2,a3] = Water[8.07131, 1730.63, 233.426]
[a1,a2,a3] = dioxine[7.43155, 1554.679, 240.337]

x1 + x2 = 1

x1 = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
p = [28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.7, 17.5]

# Formulate the Least Square Problem (Estimate A12 and A21 using data from above table)

# Since not linear, there is no analytical solution. Solve using gradient descent or Newton's method

# Compare optimized model with the data. Does your model fit well with the data






## Problem 2

min x1,x2 = (4-2.1*x1^2+(x1^4)/3)*x1^2+x1*x2+(-4+4*x2^2)*x2^2

x1 = [-3,3]
x2 = [-2,2]
# Solve using Bayesian Optimization for













