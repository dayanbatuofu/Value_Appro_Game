"""
This example is to compute the two-player general-sum differential with state penalty
V = min{\int_t^T (l+u^2) + g(x(T))}, u is control effort, l is state penalty, g is terminal loss
"""

import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *

# Specify the  file that includes dynamic systems
from odp.dynamics import Intersection
# Solver core
from odp.solver_gs import HJSolver, computeSpatDerivArray
import scipy.io as scio

import math

""" USER INTERFACES
- Define grid
- Generate initial values for grid using shape functions
- Time length for computations
- Initialize plotting option
- Call HJSolver function
"""

##################################################### Intersection Case #####################################################
# dx=1
g = Grid(np.array([15, 15, 15.1, 15]), np.array([105, 32, 105.1, 32]), 4, np.array([91, 18, 91, 18]), [])
# dx=0.5
# g = Grid(np.array([15, 15, 15.1, 15]), np.array([105, 32, 105.1, 32]), 4, np.array([181, 35, 181, 35]), [])
# dx=0.3
# g = Grid(np.array([15, 15, 15.1, 15]), np.array([105, 32, 105.1, 32]), 4, np.array([301, 56, 301, 56]), [])

# Define my object
my_car = Intersection()

# Use the grid to initialize initial value function
Initial_value1, Initial_value2 = IntersectionShape(g)
Initial_value_f = [Initial_value1, Initial_value2]

# Look-back length and time step
lookback_length = 3.0
t_step = 1.5  # default: 0.05

small_number = 1e-5

tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

# In this example, we compute a Backward Reachable Tube
result1, result2 = HJSolver(my_car, g, Initial_value_f, tau, saveAllTimeSteps=True, accuracy="high")

last_time_step_result1 = result1[..., 0]
last_time_step_result2 = result2[..., 0]

terminal_result1 = result1[..., -2]
terminal_result2 = result2[..., -2]

save_path = 'level_set_data_ana.mat'

data = {'V1': last_time_step_result1,
        'V2': last_time_step_result2}

scio.savemat(save_path, data)

# Compute spatial derivatives at every state
# d1_derivative1, d1_derivative2 = computeSpatDerivArray(g, last_time_step_result1, last_time_step_result2, deriv_dim=1, accuracy="low")
# v1_derivative1, v1_derivative2 = computeSpatDerivArray(g, last_time_step_result1, last_time_step_result2, deriv_dim=2, accuracy="low")
# d2_derivative1, d2_derivative2 = computeSpatDerivArray(g, last_time_step_result1, last_time_step_result2, deriv_dim=3, accuracy="low")
# v2_derivative1, v2_derivative2 = computeSpatDerivArray(g, last_time_step_result1, last_time_step_result2, deriv_dim=4, accuracy="low")

# d1_derivative1, d1_derivative2 = computeSpatDerivArray(g, terminal_result1, terminal_result2, deriv_dim=1, accuracy="low")
# v1_derivative1, v1_derivative2 = computeSpatDerivArray(g, terminal_result1, terminal_result2, deriv_dim=2, accuracy="low")
# d2_derivative1, d2_derivative2 = computeSpatDerivArray(g, terminal_result1, terminal_result2, deriv_dim=3, accuracy="low")
# v2_derivative1, v2_derivative2 = computeSpatDerivArray(g, terminal_result1, terminal_result2, deriv_dim=4, accuracy="low")

# Let's compute optimal control at some random idices
# spat_deriv_vector1 = (d1_derivative1[0, 3, 0, 3], v1_derivative1[0, 3, 0, 3],
#                       d2_derivative1[0, 3, 0, 3], v2_derivative1[0, 3, 0, 3])
#
# spat_deriv_vector2 = (d1_derivative2[0, 3, 0, 3], v1_derivative2[0, 3, 0, 3],
#                       d2_derivative2[0, 3, 0, 3], v2_derivative2[0, 3, 0, 3])
#
# # Compute the optimal control
# opt_a, opt_w = my_car.optCtrl_inPython(spat_deriv_vector1, spat_deriv_vector2)
# print("Optimal accel is {}\n".format(opt_a))
# print("Optimal rotation is {}\n".format(opt_w))

