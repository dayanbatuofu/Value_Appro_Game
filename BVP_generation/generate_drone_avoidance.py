'''
This script generates data from scratch using time-marching.
'''

import numpy as np
from utilities.BVP_solver import solve_bvp
import scipy.io
import time
import warnings
import copy

from utilities.other import int_input

from examples.choose_problem_drone_avoidance import system, problem, config

np.seterr(over='warn', divide='warn', invalid='warn')
warnings.filterwarnings('error')

# change the seed number you want
np.random.seed(config.random_seeds['generate'])
N_states = problem.N_states
alpha = problem.alpha

# Validation or training data?
data_type = int_input('What kind of data? Enter 0 for validation, 1 for training:')
if data_type:
    data_type = 'train'
    # data_type = 'test'
else:
    data_type = 'val'

save_path = 'examples/' + system + '/data_' + data_type + f'_drone.mat'
print(save_path)

Ns = config.Ns[data_type]
X0_pool = problem.sample_X0(Ns)


'''
If we want to separate the data into upper and lower triangle, we should use
'''
t_OUT = np.empty((1, 0))
X_OUT = np.empty((2 * N_states, 0))
A_OUT = np.empty((4 * N_states, 0))
V_OUT = np.empty((2, 0))

N_sol = 0
sol_time = []
step = 0
N_opt1 = 0
N_opt2 = 0
X0 = X0_pool[:, 0]
X0_orignal = copy.deepcopy(X0)

N_converge = 0

time_horizon = problem.t1
print(time_horizon, problem.t1)
num_iter = int(time_horizon)

# ---------------------------------------------------------------------------- #
while N_sol < Ns:
    print('Solving BVP #', N_sol + 1, 'of', Ns, '...', end=' ')

    step += 1
    X0 = X0_pool[:, N_sol]
    print(step)
    print(X0)
    bc = problem.make_bc(X0)

    start_time = time.time()
    tol = 5e-3  # 1e-01

    # Initial guess setting --> #Todo: what is this??
    X_guess = np.vstack((X0.reshape(-1, 1),
                         np.array([[alpha],
                                   [alpha],
                                   [alpha],
                                   [alpha * time_horizon],
                                   [alpha * time_horizon],
                                   [alpha * time_horizon],
                                   [0.],
                                   [0.],
                                   [0.],
                                   [0.],
                                   [0.],
                                   [0.],
                                   [0.],
                                   [0.],
                                   [0.],
                                   [0.],
                                   [0.],
                                   [0.],
                                   [alpha],
                                   [alpha],
                                   [alpha],  # alpha * 3.
                                   [alpha * time_horizon],
                                   [alpha * time_horizon],
                                   [alpha * time_horizon],
                                   [0.],
                                   [0.]])))

    X_guess1 = X_guess

    '''
    New guessing for X and t, go through all the actions and find the global solutions
    '''
    V_list = np.empty((1, 0))
    actions = [(7.81, 7.81), (11.81, 11.81), (7.81, 11.81), (11.81, 7.81)]
    X_sol = []
    Y_sol = []
    rms_sol = []
    '''
    time horizon is 3 seconds
    '''
    t_guess1 = np.linspace(0, time_horizon, num_iter+1)
    dt = t_guess1[1] - t_guess1[0]

    for i in range(num_iter):
        X_guess1 = np.hstack((X_guess1, X_guess))
    for action in actions:
        a1, a2 = action
        for i in range(int(time_horizon)):
            X_guess1[0, i + 1] = X_guess1[0, 0] + X_guess1[3, 0] * t_guess1[i + 1]
            X_guess1[1, i + 1] = X_guess1[1, 0] + X_guess1[4, 0] * t_guess1[i + 1]
            X_guess1[2, i + 1] = X_guess1[2, 0] + X_guess1[5, 0] * t_guess1[i + 1] + ((a1 - problem.g) / 2) * t_guess1[i + 1] ** 2
            X_guess1[5, i + 1] = X_guess1[5, 0] + (a1 - problem.g) * t_guess1[i + 1]
            X_guess1[6, i + 1] = X_guess1[6, 0] + X_guess1[9, i] * t_guess1[i + 1]
            X_guess1[7, i + 1] = X_guess1[7, 0] + X_guess1[10, i] * t_guess1[i + 1]
            X_guess1[8, i + 1] = X_guess1[8, 0] + X_guess1[11, 0] * t_guess1[i + 1] + ((a2 - problem.g) / 2) * t_guess1[i + 1] ** 2
            X_guess1[11, i + 1] = X_guess1[11, 0] + (a2 - problem.g) * t_guess1[i + 1]
            X_guess1[15, i + 1] = -alpha * (t_guess1[i + 1] - time_horizon)
            X_guess1[16, i + 1] = -alpha * (t_guess1[i + 1] - time_horizon)
            X_guess1[17, i + 1] = -alpha * (t_guess1[i + 1] - time_horizon)
            X_guess1[33, i + 1] = -alpha * (t_guess1[i + 1] - time_horizon)
            X_guess1[34, i + 1] = -alpha * (t_guess1[i + 1] - time_horizon)
            X_guess1[35, i + 1] = -alpha * (t_guess1[i + 1] - time_horizon)

        try:
            SOL = solve_bvp(problem.aug_dynamics, bc, t_guess1, X_guess1,
                            verbose=2, tol=tol, max_nodes=1500)
        except Exception as e:
            continue

        max_rms = np.max(SOL.rms_residuals)
        if max_rms < tol:
            V_tmp = (-SOL.y[-2:-1, 0:1]) + (-SOL.y[-1:, 0:1])
            V_list = np.hstack((V_list, V_tmp))
            X_sol.append(SOL.x)
            Y_sol.append(SOL.y)
            rms_sol.append(SOL.rms_residuals)
        else:
            pass

    if X_sol == []:
        pass
    else:
        index = np.argmax(V_list)

        t = X_sol[index]
        X = Y_sol[index][:2 * N_states]
        A = Y_sol[index][2 * N_states:6 * N_states]
        V1 = -Y_sol[index][-2:-1]
        V2 = -Y_sol[index][-1:]
        V = np.vstack((V1, V2))
        rms = np.max(rms_sol[index])

        sol_time.append(time.time() - start_time)

        t_OUT = np.hstack((t_OUT, t.reshape(1, -1)))
        X_OUT = np.hstack((X_OUT, X))
        A_OUT = np.hstack((A_OUT, A))
        V_OUT = np.hstack((V_OUT, V))

        N_converge += 1

    N_sol += 1

sol_time = np.sum(sol_time)

print('')
print(step, '/', step, 'successful solution attempts:')
print('Average solution time: %1.1f' % (sol_time / step), 'sec')
print('Total solution time: %1.1f' % (sol_time), 'sec')

print('')
print('Total data generated:', X_OUT.shape[1])
print('Converge Number:', N_converge)
print('')

# ---------------------------------------------------------------------------- #
save_data = int_input('Save data? Enter 0 for no, 1 for yes:')
if save_data:
    try:
        save_dict = scipy.io.loadmat(save_path)

        overwrite_data = int_input('Overwrite existing data? Enter 0 for no, 1 for yes:')

        if overwrite_data:
            raise RuntimeWarning

        save_dict.update({'t': np.hstack((save_dict['t'], t_OUT)),
                          'X': np.hstack((save_dict['X'], X_OUT)),
                          'A': np.hstack((save_dict['A'], A_OUT)),
                          'V': np.hstack((save_dict['V'], V_OUT))})

    except:
        Theta1, Theta2 = problem.Theta_star(np.vstack((X_OUT, A_OUT)))
        Theta = np.vstack((Theta1, Theta2))
        Phi1, Phi2 = problem.Phi_star(np.vstack((X_OUT, A_OUT)))
        Phi = np.vstack((Phi1, Phi2))
        Thrust1, Thrust2 = problem.Thrust_star(np.vstack((X_OUT, A_OUT)))
        Thrust = np.vstack((Thrust1, Thrust2))

        save_dict = {'lb_1': np.min(X_OUT[:N_states], axis=1, keepdims=True),
                     'ub_1': np.max(X_OUT[:N_states], axis=1, keepdims=True),
                     'lb_2': np.min(X_OUT[N_states:2 * N_states], axis=1, keepdims=True),
                     'ub_2': np.max(X_OUT[N_states:2 * N_states], axis=1, keepdims=True),
                     'A_lb_11': np.min(A_OUT[:N_states], axis=1, keepdims=True),
                     'A_ub_11': np.max(A_OUT[:N_states], axis=1, keepdims=True),
                     'A_lb_12': np.min(A_OUT[N_states:2 * N_states], axis=1, keepdims=True),
                     'A_ub_12': np.max(A_OUT[N_states:2 * N_states], axis=1, keepdims=True),
                     'A_lb_21': np.min(A_OUT[2 * N_states:3 * N_states], axis=1, keepdims=True),
                     'A_ub_21': np.max(A_OUT[2 * N_states:3 * N_states], axis=1, keepdims=True),
                     'A_lb_22': np.min(A_OUT[3 * N_states:4 * N_states], axis=1, keepdims=True),
                     'A_ub_22': np.max(A_OUT[3 * N_states:4 * N_states], axis=1, keepdims=True),
                     'V_min_1': np.min(V_OUT[-2:-1, :]), 'V_max_1': np.max(V_OUT[-2:-1, :]),
                     'V_min_2': np.min(V_OUT[-1, :]), 'V_max_2': np.max(V_OUT[-1, :]),
                     't': t_OUT, 'X': X_OUT, 'A': A_OUT, 'V': V_OUT, 'Theta': Theta, 'Phi': Phi, 'Thrust': Thrust}
        scipy.io.savemat(save_path, save_dict)