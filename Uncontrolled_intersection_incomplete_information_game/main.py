"""
Use python main.py to execute!
Important files:
1. autonomous_vehicle: process and record agent info
2. inference_model: performs inference and prediction
3. decision_model: returns appropriate action for each agent
4. sim_draw: plots the simulation and results
5. >>> savi_simulation: executes the simulation <<<
Change the DEFAULT values below to change actual model used!
"""
import os
import argparse
import utils
import torch as t
import numpy as np
import csv
from environment import Environment
from savi_simulation import Simulation
import shutil
import scipy.io

parser = argparse.ArgumentParser()
"""
simulation parameters
"""
parser.add_argument('--sim_duration', type=int, default=100)  # time span for simulation
parser.add_argument('--sim_dt', type=int, default=0.05)  # time step in simulation: choices: [0.01, 0.25, 1]
parser.add_argument('--sim_lr', type=float, default=0.1)  # learning rate
parser.add_argument('--sim_nepochs', type=int, default=100)  # number of training epochs
parser.add_argument('--save', type=str, default='./experiment')  # save dir
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)

"""
environment parameters
"""
parser.add_argument('--env_name', type=str, choices=['trained_intersection', 'bvp_intersection'],
                    default='bvp_intersection')
# Starting position and velocity is set within the environment

"""
agent model parameters
"""
# TODO: bvp_continuous model is under development
# choose inference model: use bvp for our experiment, and only for 1st player (i.e. ['bvp', 'none'])
parser.add_argument('--agent_inference', type=str, choices=['none', 'bvp', 'bvp_continuous'],
                    default=['bvp', 'none'])

# choose decision model: use the same model for the two agent, bvp_non_empathetic or bvp_empathetic
parser.add_argument('--agent_decision', type=str,
                    choices=['constant_speed', 'bvp_baseline',
                             'bvp_non_empathetic', 'bvp_empathetic',
                             'bvp_e_optimize', 'bvp_ne_optimize'],
                    default=['bvp_empathetic', 'bvp_empathetic'])

"""
agent parameters (for the proposed s = <x0,p0(β),β†,∆t,l>), for 2 agent case
"""

parser.add_argument('--agent_dt', type=int, default=1)  # time step in planning (NOT IN USE)
parser.add_argument('--agent_intent', type=str, choices=['NA', 'A'], default=['NA', 'NA'])  # AGENT TRUE PARAM [P1, P2]
parser.add_argument('--agent_noise', type=str, choices=['N', 'NN'], default=['NN', 'NN'])
parser.add_argument('--agent_intent_belief', type=str, choices=['NA', 'A'], default=['NA', 'NA'])  # AGENT BELIEF
parser.add_argument('--agent_noise_belief', type=str, choices=['N', 'NN'], default=['NN', 'NN'])
parser.add_argument('--belief_weight', type=float, default=0.8)

# parser.add_argument('', type=str, choices=[], default=[])
args = parser.parse_args()


if __name__ == "__main__":
    loss_table = np.empty((6, 6))  # X1 by X2 initial states table
    policy_table_1 = np.empty((6, 6))  # record the policy choice of agent (correctness)
    policy_table_2 = np.empty((6, 6))
    startpos1 = np.empty((6, 6))  # for checking if starting condition is correct
    startpos2 = np.empty((6, 6))

    save_floder = args.agent_decision[0] + ' ' + args.agent_intent[0] + '_' + args.agent_intent[1] \
                       + ' ' + 'belief' + ' ' + args.agent_intent_belief[0] + '_' + args.agent_intent_belief[1]

    simulation_dir = 'experiment/' + save_floder
    if os.path.exists(simulation_dir):
        val = input("The model directory %s exists. Overwrite? (y/n)" % simulation_dir)
        if val == 'y':
            shutil.rmtree(simulation_dir)
    os.makedirs(simulation_dir)

    """
    choose (a,a) or (na,na) to test the performance
    """
    file = 'validation_scripts/data_a_a_infer_150.mat'
    # file = 'validation_scripts/data_na_na_infer_150.mat'
    data = scipy.io.loadmat(file)
    data.update({'t0': data['t']})
    idx0 = np.nonzero(np.equal(data.pop('t0'), 0.))[1]

    X0 = data['X'][:, idx0]

    for i in range(X0.shape[1]):
        x1 = X0[0, i]
        x2 = X0[2, i]
        v1 = X0[1, i]
        v2 = X0[3, i]

        initial_states = [[x1, v1], [x2, v2]]  # x1, v1, x2, v2

        utils.makedirs(args.save)
        logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
        logger.info(args)
        device = t.device('cuda:' + str(args.gpu) if t.cuda.is_available() else 'cpu')
        close_action_set = np.linspace(-5, 10, 31)
        close_action_set = close_action_set.tolist()
        # agent choices
        if args.env_name == 'bvp_intersection':
            sim_par = {"theta": [5, 1],  # NA, A
                       "lambda": [0.1, 0.5],  # N, NN
                       "action_set": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        else:
            sim_par = {"theta": [1, 1000],
                       "lambda": [0.001, 0.005],
                       "action_set": [-8, -4, 0, 4, 8],
                       }

        e = Environment(args.env_name, args.agent_inference, sim_par, initial_states, args.sim_dt, args.agent_intent, args.agent_noise,
                        args.agent_intent_belief, args.agent_noise_belief)
        assert len(args.agent_inference) == e.N_AGENTS and len(args.agent_decision) == e.N_AGENTS

        kwargs = {"env": e,
                  "duration": args.sim_duration,
                  "n_agents": e.N_AGENTS,
                  "inference_type": args.agent_inference,
                  "decision_type": args.agent_decision,
                  "sim_dt": args.sim_dt,
                  "sim_lr": args.sim_lr,
                  "sim_par": sim_par,
                  "sim_nepochs": args.sim_nepochs,
                  "belief_weight": args.belief_weight,
                  "simulation_dir": simulation_dir}
        s = Simulation(**kwargs)
        if args.agent_inference[0] == 'bvp' or args.agent_inference[0] == 'bvp_continuous':
            loss, policy_count = s.run()
        else:
            loss = s.run()




