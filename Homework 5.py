import torch as t


# Define Functions
def f(x):
    return x[0]**2 + (x[1]-3)**2

def df(x):
    return [[2*x[0]],[2*(x[1]-3)]]

def g1(x):
    return x[1]**2 - 2*x[0]

def dg1(x):
    return [[-2],[2*x[1]]]

def g2(x):
    return (x[1]-1)**2 + 5*x[0] - 15

def dg2(x):
    return [[5],[2*(x[1]-1)]]


# Initialize Variables
err = 10**(-3)

# Initial Guess
x0 = [1,1]

## SQP Function w/ BFGS ##
def mysqp(f, df, g1, dg1, g2, dg2, x0, opt):
    x = x0

    # Initialize structure to record search process
    solution = []

    # Unitialize Hessian matrix
    W = [[1,0],[0,1]]
    mu_old = t.zeros(size(q1))
    w = t.zeros(size(g1))

    # Set termination criterion
    gnorm = t.norm(df(x) + mu_old*dg(x))

    while(gnorm > err):
        # Implement QP
        if strcmp(opt.alg, 'myqp')
            # Solve the QP subproblem to find s and mu (using your own method)
            [s, mu_new] = solveqp(x, W, df, g, dg)
        else:
            # Solve the QP subproblem to find s and mu (using MATLAB's solver)
            qpalg = optimset('Algorithm', 'active-set', 'Display', 'off');
            [s,~,~,~,lambda] = quadprog(W,[df(x)],dg(x),-g(x),[], [], [], [], [],  qpalg);
            mu_new = lambda.ineqlin

                           # opt.linesearch switches line search on or off.
                           # You can first set the variable "a" to different constant values and see how it
                           # affects the convergence.
                           if(opt.linesearch):
            [a, w] = lineSearch(f, df, g, dg, x, s, mu_old, w);
        else:
        a = 0.1;

    # Update the current solution using the step
    dx = a*s
    x = x + dx

    # Update Hessian using BFGS. Use equations (7.36), (7.73) and (7.74)
    # Compute y_k
    y_k = [df(x) + mu_new*dg(x) - df(x-dx) - mu_new*dg(x-dx)]
    # Compute theta
    if(dx*y_k >= 0.2*dx*W*dx):
        theta = 1
    else
        theta = (0.8*dx*W*dx)/(dx*W*dx-dx*y_k)

    # Compute  dg_k
    dg_k = theta*y_k + (1-theta)*W*dx
    # Compute new Hessian
    W = W + (dg_k*dg_k)/(dg_k*dx) - ((W*dx)*(W*dx))/(dx*W*dx)

    # Update termination criterion:
    gnorm = norm(df(x) + mu_new*dg(x))
    mu_old = mu_new

    # save current solution to solution.x
    solution.x = [solution.x, x]

## Linesearch on Merit Function ##
def lineSearch(f, df, g1, dg1, g2, dg2, x, s, mu_old, w_old):
    ts = .1
    b = .8
    a = 1
    D = s
    count = 0

    # Calculate weights in merit function
    w = max(abs(mu_old, .5*(w_old+abs(mu_old))))

    while(count<100):
        # Calculate phi using merit function
        phi_a = f(x + a*D) + w*abs(min(0, -g1(x+a*D))) + w*abs(min(0, -g2(x+a*D)))

        # Calculate psi ising phi
        phi0 = f(x) + w*abs(min(0,-g1(x))) + w*abs(min(0,-g2(x)))
        dphi0 = df(x)*D + w*((dg1(x)*D)*(g1(x)>0)) + w*((dg2(x)*D)*(g2(x)>0))
        psi_a = phi0 + ts*a*dphi0

        # Stop if condition is satisfied
        if(phi_a < psi_a):
            break
        else:
            a = a*b
            count = count + 1

    return [a, w]

## QP Subproblem ##
def solveqp(x, W, df, g1, dg1, g2, dg2):
    # min (1/2)*s'*W*s + c'*s
    # st  A*s-b <= 0

    # Compute values for problem formulation
    c = [df(x)]
    A0 = dg1(x)
    B0 = -g(x)

    #Initialize active set variables
    stop = 0
    A = []
    B = []
    active = []

    while(stop = 0):
        mu0 = t.zeros(size(g(x)))

        # Extract working set
        A = A0(active, :)
        B = B0(active)

        # Solve the QP problem
        [s, mu] = solve_activeset(x, W, c, A, b);

        # Round mu to prevent numerical errors (Keep this)
        mu = round(mu*1e12)/1e12

        # Update value for working set
        mu0(active) = mu

        # Round constraint values to prevent numerical errors (Keep this)
        gcheck = round(gcheck*1e12)/1e12

        # Variable to check if all mu values make sense.
        mucheck = 0

        # Indices of the constraints to be added to the working set
        Iadd = []
        # Indices of the constraints to be added to the working set
        Iremove = []

        # Check mu values and set mucheck to 1 when they make sense
        if (numel(mu) == 0):
            # When there no mu values in the set
            mucheck = 1
        elseif(min(mu) > 0):
        # When all mu values in the set positive
        mucheck = 1
    else:
        # When some of the mu are negative
        # Find the most negaitve mu and remove it from acitve set
        [~,Iremove] = min(mu)

    # Check if constraints are satisfied
    if(max(gcheck) <= 0):
        if(mucheck == 1):
            stop = 1;
        end
    else:
        # Find the most violated one and add it to the working set
        [~,Iadd] = max(gcheck); % Use Iadd to add the constraint

    # Remove the index Iremove from the working-set
    active = setdiff(active, active(Iremove))
    # Add the index Iadd to the working-set
    active = [active, Iadd]

    # Make sure there are no duplications in the working-set (Keep this)
    active = unique(active)


def solve_activeset(x, W, c, A, b):
    # Given an active set, solve QP

    # Create the linear set of equations given in equation (7.79)
    M = [[W, A], [A, zeros(size(A,1))]]
    U = [[-c], [b]]
    sol = M\U;          % Solve for s and mu

    s = sol(1:numel(x))                # Extract s from the solution
    mu = sol(numel(x)+1:numel(sol))    # Extract mu from the solution










