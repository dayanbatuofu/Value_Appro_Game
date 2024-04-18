"""
for obtaining action for each agent
"""
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import Bounds, minimize, minimize_scalar
from validation_scripts.Hamilton_generation import get_Hamilton
import dynamics

class DecisionModel:
    def __init__(self, model, sim):
        self.sim = sim
        self.frame = self.sim.frame
        self.time = self.sim.time
        self.dt = self.sim.dt
        if model == 'constant_speed':
            self.plan = self.constant_speed
        elif model == 'bvp_baseline':  # for testing the value network by iterating through actions
            self.plan = self.bvp_baseline
        elif model == 'bvp_non_empathetic':  # using BVP value network
            self.plan = self.bvp_non_empathetic
        elif model == 'bvp_empathetic':  # using BVP value network
            self.plan = self.bvp_empathetic
        elif model == 'bvp_e_optimize':
            self.plan = self.bvp_e_optimize
        elif model == 'bvp_ne_optimize':
            self.plan = self.bvp_ne_optimize
        else:
            # placeholder for future development
            print("WARNING!!! NO DECISION MODEL FOUND")
            pass

        self.policy_or_Q = 'Q'  # for nfsp
        self.noisy = False  # if baseline is randomly picking action based on distribution (for nfsp baseline)

        self.true_params = self.sim.true_params
        self.belief_params = self.sim.initial_belief
        self.action_set = self.sim.action_set
        self.theta_list = self.sim.theta_list
        self.lambda_list = self.sim.lambda_list
        self.beta_set = self.sim.beta_set

    @staticmethod
    def constant_speed():
        return {'action': 0}  # just keep the speed

    def bvp_baseline(self):
        """
        Choose action according to given intent, using the BVP value approximated model
        :return:
        """
        "sorting states to obtain action from pre-trained model"
        # y direction only for M, x direction only for HV
        self.frame = self.sim.frame
        self.time = self.sim.time
        p1_state = self.sim.agents[0].state[self.sim.frame]
        p2_state = self.sim.agents[1].state[self.sim.frame]

        lambda_list = self.lambda_list
        theta_list = self.theta_list
        action_set = self.action_set

        "Baseline: get self true betas from initial setting"
        true_beta_h, true_beta_m = self.true_params

        "METHOD 1: Get the whole p_action table using true param of self and other"
        "METHOD 2: Get p_action based on only the last action observed"
        actions = []

        for i in range(self.sim.n_agents):
            last_a_other = self.sim.agents[i - 1].action[self.frame - 1]  # other agent's last action
            p_action_i = self.bvp_action_prob_2(i, p1_state, p2_state, true_beta_h, true_beta_m, last_a_other)
            actions.append(action_set[np.argmax(p_action_i)])

        # print("action taken for baseline:", actions, "current state (y is reversed):", p1_state, p2_state)
        return {'action': actions}

    def bvp_optimize_global(self):
        """
        Uses scipy optimizer to find best action for baseline test
        :return:
        """
        "sorting states to obtain action from pre-trained model"
        # y direction only for M, x direction only for HV
        self.frame = self.sim.frame
        self.time = self.sim.time
        dt = self.dt
        p1_state = self.sim.agents[0].state[self.sim.frame]
        p2_state = self.sim.agents[1].state[self.sim.frame]
        lambda_list = self.lambda_list
        theta_list = self.theta_list
        action_set = self.action_set
        "Baseline: get self true betas from initial setting"
        true_beta_h, true_beta_m = self.true_params
        theta_1 = true_beta_h[0]
        theta_2 = true_beta_m[0]
        # x_nn should be new x after the action
        time = np.array([[self.time]])
        last_a_1 = self.sim.agents[0].last_actions[-1][0]
        last_a_2 = self.sim.agents[0].last_actions[-1][1]
        new_p1_s_last = dynamics.bvp_dynamics_1d(p1_state, last_a_1, dt)  # resulting state from action observed
        new_p2_s_last = dynamics.bvp_dynamics_1d(p2_state, last_a_2, dt)
        x1_nn = np.array([[p1_state[1]], [p1_state[3]], [p2_state[0]], [p2_state[2]]])

        def q_function_helper1(action):  # for agent 1
            q1, q2 = get_Hamilton(x1_nn, time, np.array([[action], [last_a_2]]),
                                  (true_beta_h[0], true_beta_m[0]))
            q1 = -q1[0][0]
            print(action, q1)
            return q1

        def q_function_helper2(action):  # for agent 2
            q1, q2 = get_Hamilton(x1_nn, time, np.array([[last_a_1], [action]]),
                                 (true_beta_h[0], true_beta_m[0]))
            q2 = -q2[0][0]
            print(action, q2)
            return q2

        res1 = minimize_scalar(q_function_helper1)  # for agent 1
        res2 = minimize_scalar(q_function_helper2)
        res = [res1.x, res2.x]
        actions = []
        for a in res:
            if a > 10:
                actions.append(10)
            elif a < -5:
                actions.append(-5)
            else:
                actions.append(a)

        assert len(actions) == 2
        return {'action': actions}

    def bvp_non_empathetic(self):
        """
        Uses BVP value network
        Get appropriate action based on predicted intent of the other agent and self intent
        :return: appropriate action for both agents
        """
        # implement reactive planning based on point estimates of future trajectories

        self.frame = self.sim.frame
        self.time = self.sim.time
        "sorting states to obtain action from pre-trained model"
        # y direction only for M, x direction only for HV
        p1_state = self.sim.agents[0].state[self.sim.frame]
        p2_state = self.sim.agents[1].state[self.sim.frame]
        action_set = self.action_set

        "this is where non_empathetic is different: using true param of self to observe portion of belief table"
        if self.sim.frame == 0:
            p_beta = self.sim.initial_belief
        else:
            p_beta, ne_betas = self.sim.agents[0].predicted_intent_all[-1]

        "true_param_id -> get row/col of p_beta -> get predicted beta"
        true_beta_1, true_beta_2 = self.true_params
        b_id_1 = self.beta_set.index(true_beta_1)
        b_id_2 = self.beta_set.index(true_beta_2)
        p_b_1 = np.transpose(p_beta)[b_id_2]  # get col p_beta
        p_b_2 = p_beta[b_id_1]
        beta_1 = self.beta_set[np.argmax(p_b_1)]
        beta_2 = self.beta_set[np.argmax(p_b_2)]

        "Get p_action based on only the last action observed"
        actions = []
        p_a_2 = []
        for i in range(self.sim.n_agents):
            last_a_other = self.sim.agents[0].last_actions[-1][i-1]  # other agent's last action
            if i == 0:
                p_action_i = self.bvp_action_prob_2(i, p1_state, p2_state, true_beta_1, beta_2, last_a_other)
            elif i == 1:
                p_action_i = self.bvp_action_prob_2(i, p1_state, p2_state, beta_1, true_beta_2, last_a_other)
            p_a_2.append(p_action_i)
            actions.append(action_set[np.argmax(p_action_i)])

        # print("action taken:", actions, "current state (y is reversed):", p1_state, p2_state)

        return {'action': actions}

    def bvp_empathetic(self):
        """
        Choose action from Nash Equilibrium, according to the inference model
        :return: actions for both agents
        """
        # implement reactive planning based on inference of future trajectories
        self.frame = self.sim.frame
        self.time = self.sim.time

        "sorting states to obtain action from pre-trained model"
        # y direction only for M, x direction only for HV
        p1_state = self.sim.agents[0].state[self.sim.frame]
        p2_state = self.sim.agents[1].state[self.sim.frame]
        p1_state_nn = (p1_state[1], p1_state[3], p2_state[0], p2_state[2])  # s_ego, v_ego, s_other, v_other
        p2_state_nn = ([p2_state[0]], [p2_state[2]], [p1_state[1]], [p1_state[3]])

        lambda_list = self.lambda_list
        theta_list = self.theta_list
        action_set = self.action_set

        "this is where empathetic is different: using predicted param of self with entire belief table"
        if self.sim.frame == 0:
            p_beta = self.sim.initial_belief
            beta_pair_id = np.unravel_index(p_beta.argmax(), p_beta.shape)
            beta_h = self.beta_set[beta_pair_id[0]]
            beta_m = self.beta_set[beta_pair_id[1]]
        else:
            p_beta, [beta_h, beta_m] = self.sim.agents[0].predicted_intent_all[-1]
        true_beta_h, true_beta_m = self.true_params

        "Get p_action based on only the last action observed"
        actions = []
        p_a_2 = []  # for comparison
        for i in range(self.sim.n_agents):
            last_a_other = self.sim.agents[0].last_actions[-1][i-1]  # other agent's last action
            if i == 0:
                p_action_i = self.bvp_action_prob_2(i, p1_state, p2_state, true_beta_h, beta_m, last_a_other)
            elif i == 1:
                p_action_i = self.bvp_action_prob_2(i, p1_state, p2_state, beta_h, true_beta_m, last_a_other)
            p_a_2.append(p_action_i)
            actions.append(action_set[np.argmax(p_action_i)])

        # print("action taken:", actions, "current state (y is reversed):", p1_state, p2_state)
        # actions = [action1, action2]
        return {'action': actions}

    def bvp_e_optimize(self):
        """
        BVP empathetic
        Uses scipy optimizer to find best action
        :return:
        """
        "sorting states to obtain action from pre-trained model"
        # y direction only for M, x direction only for HV
        self.frame = self.sim.frame
        self.time = self.sim.time
        dt = self.dt
        p1_state = self.sim.agents[0].state[self.sim.frame]
        p2_state = self.sim.agents[1].state[self.sim.frame]
        # p1_state_nn = (p1_state[1], p1_state[3], p2_state[0], p2_state[2])  # s_ego, v_ego, s_other, v_other
        # p2_state_nn = (p2_state[0], p2_state[2], p1_state[1], p1_state[3])
        lambda_list = self.lambda_list
        theta_list = self.theta_list
        action_set = self.action_set

        "this is where empathetic is different: using predicted param of self with entire belief table"
        if self.sim.frame == 0:  # get prob beta pair from initially generated one
            p_beta = self.sim.initial_belief
            beta_pair_id = np.unravel_index(p_beta.argmax(), p_beta.shape)
            est_beta_1 = self.beta_set[beta_pair_id[0]]
            est_beta_2 = self.beta_set[beta_pair_id[1]]
            est_theta_1 = est_beta_1[0]
            est_theta_2 = est_beta_2[0]
        else:
            p_beta, [est_beta_1, est_beta_2] = self.sim.agents[0].predicted_intent_all[-1]
            est_theta_1 = est_beta_1[0]
            est_theta_2 = est_beta_2[0]
        true_beta_1, true_beta_2 = self.true_params

        time = np.array([[self.time]])
        last_a_1 = self.sim.agents[0].last_actions[-1][0]
        last_a_2 = self.sim.agents[0].last_actions[-1][1]
        "getting new states (if needed by NN)"
        x1_nn = np.array([[p1_state[1]], [p1_state[3]], [p2_state[0]], [p2_state[2]]])

        def q_function_helper1(action):  # for agent 1
            """Helper function for optimization (agent 1)"""
            # new_p1_s = dynamics.bvp_dynamics_1d(p1_state, action, dt)
            # new_x1_nn = np.array([[new_p1_s[1]], [new_p1_s[3]], [new_p2_s_last[0]], [new_p2_s_last[2]]])
            "Hamiltonian from costate network"
            # q1, q2 = get_Hamilton(x1_nn, time, np.array([[action], [last_a_2]]),
            #                      (true_beta_h[0], true_beta_m[0]))
            "Q value network"
            q1, q2 = get_Hamilton(x1_nn, time, np.array([[action], [last_a_2]]),
                                 (true_beta_1[0], est_beta_2[0]))
            q1 = -q1[0][0]
            print(action, q1)
            return q1

        def q_function_helper2(action):  # for agent 2
            """Helper function for optimization (agent 2)"""
            # new_p2_s = dynamics.bvp_dynamics_1d(p2_state, action, dt)
            # new_x1_nn = np.array([[new_p1_s_last[1]], [new_p1_s_last[3]], [new_p2_s[0]], [new_p2_s[2]]])
            "Q value network"
            q1, q2 = get_Hamilton(x1_nn, time, np.array([[last_a_1], [action]]),
                                 (true_beta_1[0], est_beta_2[0]))
            "Hamiltonian from costate network"
            # q1, q2 = get_Hamilton(x1_nn, time, np.array([[last_a_1], [action]]),
            #                      (true_beta_h[0], true_beta_m[0]))
            q2 = -q2[0][0]
            print(action, q2)
            return q2

        "scipy minimizer: globally minimizing -Q (maximizing Q)"
        res1 = minimize_scalar(q_function_helper1)  # , bounds=(-5, 10), method="bounded")
        res2 = minimize_scalar(q_function_helper2)  # , bounds=(-5, 10), method="bounded")
        res = [res1.x, res2.x]
        actions = []
        for a in res:
            if a > 10:
                actions.append(10)
            elif a < -5:
                actions.append(-5)
            else:
                actions.append(a)

        assert len(actions) == 2

        return {'action': actions}

    def bvp_ne_optimize(self):
        """
        BVP non-empathetic
        Uses scipy optimizer to find best action
        :return:
        """
        "sorting states to obtain action from pre-trained model"
        # y direction only for M, x direction only for HV
        self.frame = self.sim.frame
        self.time = self.sim.time
        dt = self.dt
        p1_state = self.sim.agents[0].state[self.sim.frame]
        p2_state = self.sim.agents[1].state[self.sim.frame]
        lambda_list = self.lambda_list
        theta_list = self.theta_list
        action_set = self.action_set

        "this is where non_empathetic is different: using true param of self to observe portion of belief table"
        if self.sim.frame == 0:
            p_beta = self.sim.initial_belief
        else:
            p_beta, ne_betas = self.sim.agents[0].predicted_intent_all[-1]

        "true_param_id -> get row/col of p_beta -> get predicted beta"
        true_beta_1, true_beta_2 = self.true_params
        b_id_1 = self.beta_set.index(true_beta_1)
        b_id_2 = self.beta_set.index(true_beta_2)
        p_b_1 = np.transpose(p_beta)[b_id_2]  # get col p_beta
        p_b_2 = p_beta[b_id_1]
        est_beta_1 = self.beta_set[np.argmax(p_b_1)]
        est_beta_2 = self.beta_set[np.argmax(p_b_2)]
        est_theta_1 = est_beta_1[0]
        est_theta_2 = est_beta_2[0]

        time = np.array([[self.time]])
        last_a_1 = self.sim.agents[0].last_actions[-1][0]
        last_a_2 = self.sim.agents[0].last_actions[-1][1]

        "getting new states (if needed by NN)"
        x1_nn = np.array([[p1_state[1]], [p1_state[3]], [p2_state[0]], [p2_state[2]]])

        def q_function_helper1(action):  # for agent 1
            # new_p1_s = dynamics.bvp_dynamics_1d(p1_state, action, dt)
            # new_x1_nn = np.array([[new_p1_s[1]], [new_p1_s[3]], [new_p2_s_last[0]], [new_p2_s_last[2]]])
            # q1, q2 = get_Hamilton(x1_nn, time, np.array([[action], [last_a_2]]),
            #                      (true_beta_h[0], true_beta_m[0]))
            q1, q2 = get_Hamilton(x1_nn, time, np.array([[action], [last_a_2]]),
                                 (true_beta_1[0], est_beta_2[0]))
            q1 = -q1[0][0]
            print(action, q1)
            return q1

        def q_function_helper2(action):  # for agent 2
            # new_p2_s = dynamics.bvp_dynamics_1d(p2_state, action, dt)
            # new_x1_nn = np.array([[new_p1_s_last[1]], [new_p1_s_last[3]], [new_p2_s[0]], [new_p2_s[2]]])
            q1, q2 = get_Hamilton(x1_nn, time, np.array([[last_a_1], [action]]),
                                 (true_beta_1[0], est_beta_2[0]))
            # q1, q2 = get_Hamilton(x1_nn, time, np.array([[last_a_1], [action]]),
            #                      (true_beta_h[0], true_beta_m[0]))
            q2 = -q2[0][0]
            print(action, q2)
            return q2

        "scipy minimizer: globally minimizing -Q (maximizing Q)"
        res1 = minimize_scalar(q_function_helper1)  # , bounds=(-5, 10), method="bounded")
        res2 = minimize_scalar(q_function_helper2)  # , bounds=(-5, 10), method="bounded")
        res = [res1.x, res2.x]
        actions = []
        for a in res:
            if a > 10:
                actions.append(10)
            elif a < -5:
                actions.append(-5)
            else:
                actions.append(a)

        assert len(actions) == 2

        return {'action': actions}

    "-------------Utilities:---------------"
    def bvp_action_prob_2(self, id, p1_state, p2_state, beta_1, beta_2, last_action):
        """
        faster version than bvp_p_action
        calculate action prob for one agent: simplifies the calculation
        :param p1_state:
        :param p2_state:
        :param _lambda:
        :param theta:
        :return: p_action of p_i, where p_action = [p_a1, ..., p_a5]
        """

        theta_1, lambda_1 = beta_1
        theta_2, lambda_2 = beta_2
        action_set = self.action_set
        _p_action = np.zeros((len(action_set)))  # 1D

        time = np.array([[3 - self.time]])
        "Need state for agent H: xH, vH, xM, vM"
        "Here it does not need to compute the state again because it is the current state"
        if id == 0:
            for i, p_a_h in enumerate(_p_action):
                p_state_nn = np.array([[p1_state[1]], [p1_state[3]], [p2_state[0]], [p2_state[2]]])
                q1, q2 = get_Hamilton(p_state_nn, time, np.array([[action_set[i]], [last_action]]),
                                      (theta_1, theta_2))
                _p_action[i] = q1 * lambda_1
        elif id == 1:  # p2
            for i, p_a_h in enumerate(_p_action):
                p_state_nn = np.array([[p1_state[1]], [p1_state[3]], [p2_state[0]], [p2_state[2]]])
                q1, q2 = get_Hamilton(p_state_nn, time, np.array([[last_action], [action_set[i]]]),
                                      (theta_1, theta_2))
                _p_action[i] = q2 * lambda_2
        else:
            print("WARNING! AGENT COUNT EXCEEDED 2")

        "using logsumexp to prevent nan"
        Q_logsumexp = logsumexp(_p_action)
        "normalizing"
        _p_action -= Q_logsumexp
        _p_action = np.exp(_p_action)

        print("action prob 1 from bvp:", _p_action)
        assert round(np.sum(_p_action)) == 1

        return _p_action  # exp_Q





