"""
Environment class: create agent information/parameters and initial states
"""
import numpy as np
import savi_simulation as sim
import dynamics
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar, minimize

class Environment:
    # add: intent, noise, intent belief, noise belief
    def __init__(self, env_name, agent_inference, sim_par, initial_states, sim_dt, agent_intent, agent_noise, agent_intent_belief, agent_noise_belief):

        self.name = env_name
        self.sim = sim
        self.sim_par = sim_par
        self.dt = sim_dt
        self.agent_intent = []
        self.agent_noise = []
        self.agent_intent_belief = []
        self.agent_noise_belief = []
        self.initial_states = initial_states
        self.agent_inference = agent_inference[0]

        for i in range(len(agent_intent)):
            'check agent theta'
            if agent_intent[i] == 'NA':
                self.agent_intent.append(sim_par['theta'][0])
            elif agent_intent[i] == 'A':
                self.agent_intent.append(sim_par['theta'][1])
            'check agent theta belief'
            if agent_intent_belief[i] == 'NA':
                self.agent_intent_belief.append(sim_par['theta'][0])
            elif agent_intent_belief[i] == 'A':
                self.agent_intent_belief.append(sim_par['theta'][1])
            'check agent lambda'
            if agent_noise[i] == 'N':
                self.agent_noise.append(sim_par['lambda'][0])
            elif agent_noise[i] == 'NN':
                self.agent_noise.append(sim_par['lambda'][1])
            'check agent lambda belief'
            if agent_noise_belief[i] == 'N':
                self.agent_noise_belief.append(sim_par['lambda'][0])
            elif agent_noise_belief[i] == 'NN':
                self.agent_noise_belief.append(sim_par['lambda'][1])

        if self.name == 'bvp_intersection':
            self.N_AGENTS = 2
            self.CAR_WIDTH = 1.5  # m
            self.CAR_LENGTH = 3  # m
            self.MIN_SPEED = 0.1
            self.MAX_SPEED = 30

            # # BOUNDS: [agent1, agent2, ...], agent: [bounds along x, bounds along y], bounds: [min, max]
            # self.bounds = [[[-boundx, boundx], None], [None, [-boundy, boundy]]]
            self.bounds = [[[-self.CAR_WIDTH / 2, self.CAR_WIDTH / 2], None],
                           [None, [-self.CAR_WIDTH / 2, self.CAR_WIDTH / 2]]]
            # first car (H) moves bottom up, second car (M) right to left

            "randomly pick initial states:"
            # initial state range: x: 15 to 20, v: 18 to 25
            # u range: [-5 10]
            sy_H = self.initial_states[0][0]  # P1
            vy_H = self.initial_states[0][1]
            sx_M = self.initial_states[1][0]  # P2
            vx_M = self.initial_states[1][1]

            assert 20 >= sy_H >= 15
            assert 20 >= sx_M >= 15
            self.car_par = [{"sprite": "grey_car_sized.png",
                             "initial_state": [[0, sy_H, 0, vy_H]],  # pos_x, pos_y, vel_x, vel_y
                             "desired_state": [0, 0.4],  # pos_x, pos_y
                             "initial_action": [],  # accel
                             "par": [self.agent_intent[0], self.agent_noise[0]],  # DON'T CHANGE; par is defined in main
                             "belief": [self.agent_intent_belief[0], self.agent_noise_belief[0]],
                             # belief of other's params (beta: (theta, lambda))
                             "orientation": 0.},
                            {"sprite": "white_car_sized.png",
                             "initial_state": [[sx_M, 0, vx_M, 0]],
                             "desired_state": [-0.4, 0],
                             "initial_action": [],
                             "par": [self.agent_intent[1], self.agent_noise[1]],  # aggressiveness: check main
                             "belief": [self.agent_intent_belief[1], self.agent_noise_belief[1]],
                             # belief of other's params (beta: (theta, lambda))
                             "orientation": -90.},
                            ]

            "choose action base on decision type and intent"
            action_set = self.sim_par["action_set"]
            "METHOD 1: Get the whole p_action table using true param of self and belief of other's param"
            p1_state = self.car_par[0]["initial_state"][0]
            p2_state = self.car_par[1]["initial_state"][0]
            true_beta_h = self.car_par[0]["par"]
            true_beta_m = self.car_par[1]["par"]
            belief_beta_h = self.car_par[1]["belief"]
            belief_beta_m = self.car_par[0]["belief"]

            # TODO: solution to initial actions?
            if self.agent_inference == 'bvp':
                action_1 = 0  # placeholders!
                action_2 = 0
                self.car_par[0]["initial_action"] = [action_1]
                self.car_par[1]["initial_action"] = [action_2]
            # "METHOD 2: use optimizer to obtain the action (minimize q1+q2)"
            elif self.agent_inference == 'bvp_continuous':
                action_1 = 0  # placeholders!
                action_2 = 0
                self.car_par[0]["initial_action"] = [action_1]
                self.car_par[1]["initial_action"] = [action_2]
            else:  # none inference, etc
                action_1 = 0  # placeholders!
                action_2 = 0
                self.car_par[0]["initial_action"] = [action_1]
                self.car_par[1]["initial_action"] = [action_2]
                # print("WARNING!!!INFERENCE MODEL NOT SUPPORTED!")
                # assert len(self.car_par[0]["initial_action"]) != 0

        elif self.name == 'lane_change':
            pass

        else:
            pass


