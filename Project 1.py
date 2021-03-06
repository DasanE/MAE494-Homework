# Import proper libraries
import logging
import math
import random
import numpy as np
import time
import torch as t
import torch.nn as nn
from torch import optim
from torch.nn import utils
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Environment parameters
FRAME_TIME = 0.1  # time interval
GRAVITY_ACCEL = 0.12  # gravity constant
VERT_BOOST_ACCEL = 0.18  # vertical thrust constant
THETA_BOOST_ACCEL = .14 # angular thrust constant
DRAG_ACCEL = 0.1 # drag constant

# Rocket Data
ROCKET_WIDTH = 1 # width of the rocket
ROCKET_HEIGHT = 4 # height of the rocket
RANDOMNESS = .000 # importance of the randomness
BATCH_SIZE = 5 # number of initial states being used to train

# Error Weights
y_weight = 1
y_dot_weight = 2
theta_weight = 1
theta_dot_weight = 2


class Dynamics(nn.Module):

    def __init__(self):
        super(Dynamics, self).__init__()

    @staticmethod
    def forward(state, action):

        """
        action1: vert thrust or no thrust
        action2: clockwise angular thrust or no thrust
        action3: counterclockwise angular thrust or not thrust
        state[0] = y
        state[1] = y_dot
        state[2] = theta
        state[3] = theta_dot
        """

        # Apply gravity
        delta_state_gravity = t.tensor([0., GRAVITY_ACCEL * FRAME_TIME, 0., 0.])

        # Apply vertical thrust
        delta_state_thrust = VERT_BOOST_ACCEL * FRAME_TIME * t.tensor([0., -1., 0., 0.]) * action[0]

        # Apply drag
        ROCKET_AREA = ROCKET_WIDTH*t.cos(t.tensor([0., 0., 1., 0.])*state)+ROCKET_HEIGHT*t.sin(t.tensor([0., 0., 1., 0.])*state)
        delta_state_drag = DRAG_ACCEL * FRAME_TIME * t.tensor([0., -1., 0., 0.]) * ROCKET_AREA * (t.tensor([0., -1., 0., 0.])*state) ** 2

        # Apply angular thrust
        c_delta_state_angle = THETA_BOOST_ACCEL * FRAME_TIME * t.tensor([0., 0., 0., -1.]) * action[1]
        cc_delta_state_angle = THETA_BOOST_ACCEL * FRAME_TIME * t.tensor([0., 0., 0., 1.]) * action[2]

        # Add randomn events
        Rand = t.rand(4)
        Rand[0] = 0
        Rand[2] = 0
        Polarity = t.rand(1)
        if(Polarity < .5):
            Rand = Rand * -1
        delta_state_rand = Rand * RANDOMNESS

        # Update velocity
        state = state + delta_state_thrust + delta_state_gravity + delta_state_drag + delta_state_rand + c_delta_state_angle + cc_delta_state_angle

        # Update state
        step_mat = t.tensor([[1., FRAME_TIME, 0., 0.],[0., 1., 0., 0.],[0., 0., 1., FRAME_TIME],[0., 0., 0., 1.]])
        state = t.matmul(step_mat, state)
        return state


class Controller(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_output):
        """
        dim_input: # of system states
        dim_output: # of actions
        dim_hidden: up to you
        """

        # nn.Sigmoid()
        # nn.Tanh()
        # nn.ReLU()

        super(Controller, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(), nn.Sigmoid(),
            nn.Linear(dim_hidden, dim_output),
            nn.Tanh()
        )


    def forward(self, state):
        action = self.network(state)
        return action


class Simulation(nn.Module):
    # runs through the steps (T)
    def __init__(self, controller, dynamics, T, i):
        super(Simulation, self).__init__()
        self.state = self.initialize_state(i)
        self.controller = controller
        self.dynamics = dynamics
        self.T = T
        self.action_trajectory = []
        self.state_trajectory = []

    def forward(self, state):
        self.action_trajectory = []
        self.state_trajectory = []
        for _ in range(T):
            action = self.controller.forward(state)
            state = self.dynamics.forward(state, action)
            self.action_trajectory.append(action)
            self.state_trajectory.append(state)
        return self.error(state)

    @staticmethod
    def initialize_state(i):
        batch = [[1.3, 0., -5.6, 1.],[4.2, -.5, -3.2, 0.],[2.1, 0., 6.1, -1.8],[6.25, -0.75, -2.7, 0.5],[2.5, 1.1, 5.6, 2.8]]
        state = batch[i]
        print("The initial state being tested = ", state)
        return t.tensor(state, requires_grad=False).float()

    def error(self, state):
        obj = y_weight*state[0]**2 + y_dot_weight*state[1]**2 + theta_weight*state[2]**2 + theta_dot_weight*state[3]**2
        return obj


class Optimize:
    # calculates the gradient and the loss to optimize the neural network
    def __init__(self, simulation):
        self.simulation = simulation
        self.parameters = simulation.controller.parameters()
        self.optimizer = optim.LBFGS(self.parameters, lr=0.01)

    def step(self):
        def closure():
            loss = self.simulation(self.simulation.state)
            self.optimizer.zero_grad()
            loss.backward()
            return loss
        self.optimizer.step(closure)
        return closure()

    def train(self, epochs):
        for epoch in range(epochs):
            loss = self.step()
            print('[%d] loss: %.3f' % (epoch + 1, loss))
            if((epoch+1) == epochs):
                self.visualize()

    def visualize(self):
        data = np.array([self.simulation.state_trajectory[i].detach().numpy() for i in range(self.simulation.T)])
        # Plots the position of the rocket with respect to velocity over time
        y = data[:, 0]
        y_dot = data[:, 1]
        theta = data[:, 2]
        theta_dot = data[:, 3]

        # Plots the data as a subplot
        fig = plt.figure()

        # plots the position vs the velocity of the rocket
        plt.subplot(1,2,1)
        plt.xlabel("Height of Rocket")
        plt.ylabel("Velocity of Rocket")
        plt.plot(y, y_dot)

        # plots the angle of the rocket and its rotational velocity
        plt.subplot(1,2,2)
        plt.plot(theta,theta_dot)
        plt.xlabel("Angle of Rocket")
        plt.ylabel("Angle Velocity of Rocket")
        plt.show()

    # Now it's time to run the code!

T = 100  # number of time steps
dim_input = 4  # state space dimensions
dim_hidden = 12  # latent dimensions
dim_output = 3  # action space dimensions
d = Dynamics()  # define dynamics
c = Controller(dim_input, dim_hidden, dim_output)  # define controller

# trains the neural network over a batch of initial states
for i in range(BATCH_SIZE):
    s = Simulation(c, d, T, i)  # define simulation
    o = Optimize(s)  # define optimizer
    print("System has been initialized")
    o.train(40)  # solve the optimization problem
    print("Training for this initial state is complete")






