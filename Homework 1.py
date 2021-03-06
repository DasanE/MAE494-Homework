## Homework 1
## Dasan

# Problem 1
# Minimize -> (x1-x2)^2 + (x2+x3-2)^2 + (x4-1)^2 + (x5-1)^2
# Subject to -> x1 + 3x2 = 0
#               x3 + x4 -2x5 = 0
#               x2 - x5 = 0
#               -10 < xi < 10      ~i = 1 ... 5

from scipy.optimize import minimize

fun = lambda x: (x[0]-x[1])**2 + (x[1]+x[2]-2)**2 + (x[3]-1)**2 + (x[4]-1)**2
const = ({'type': 'eq', 'fun': lambda x: x[0]+3*x[1]},
         {'type': 'eq', 'fun': lambda x: x[2]+x[3]-2*x[4]},
         {'type': 'eq', 'fun': lambda x: x[1]-x[4]})
bounds = ((-10,10),(-10,10),(-10,10),(-10,10),(-10,10))
res = minimize(fun, (-9,3,5,-7,9), method = 'SLSQP', bounds = bounds, constraints = const)

print(res)