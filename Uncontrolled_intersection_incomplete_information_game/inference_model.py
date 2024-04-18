"""
Perform inference on other agent, updates the common belief table
In use:
    No_inference: agent does not infer other agent's param
    BVP_empathetic: uses Value network from BVP solver to evaluate agent's model
test models:
    Test_baseline: using the most basic Q function estimation, on how fast goal is reached
          (need to change main.py, environment = intersection)
    Trained_baseline: using NFSP model as Q function (change environment = trained_intersection)
    Empathetic: with NFSP, perform inference on both agent (change environment = trained_intersection)
"""
import numpy as np
from scipy.special import logsumexp
from validation_scripts.Hamilton_generation import get_Hamilton
import dynamics
from scipy.integrate import quad
from scipy.optimize import Bounds, minimize, minimize_scalar

class InferenceModel:
    def __init__(self, model, sim):  # model = inference type, sim = simulation class
        if model == 'none':
            self.infer = self.no_inference
        elif model == 'bvp':  # use with bvp decision models and env
            self.infer = self.bvp_inference
        elif model == 'bvp_continuous':
            self.infer = self.bvp_opti_inference
        else:
            # placeholder for future development
            pass

        "---simulation info:---"
        self.sim = sim
        self.agents = sim.agents
        self.n_agents = sim.env.N_AGENTS
        self.frame = sim.frame
        self.time = sim.time
        self.T = 1  # one step look ahead/ Time Horizon
        self.dt = sim.dt  # default is 1s: assigned in main.py
        self.car_par = sim.env.car_par
        self.min_speed = self.sim.env.MIN_SPEED
        self.max_speed = self.sim.env.MAX_SPEED
        "---goal states (for baseline)---"
        self.goal = [self.car_par[0]["desired_state"], self.car_par[1]["desired_state"]]

        "---parameters(theta and lambda)---"
        self.lambda_list = self.sim.lambda_list
        self.theta_list = self.sim.theta_list

        "----for BVP empathetic inference:----"
        self.beta_initial_belief = self.sim.initial_belief
        self.p_betas_prior = self.sim.initial_belief  # this is updated over time
        self.q2_prior = None
        self.past_beta = []
        self.beta_set = self.sim.beta_set
        self.action_pair_score = []
        self.belief_count = np.zeros((len(self.beta_set), len(self.beta_set)))
        self.policy_choice = [[], []]  # 0 or not correct, 1 for correct guesses (for 2 agents)

        "---Params for belief calculation (for test baseline, single agent)---"
        self.initial_belief = None  # p0: initial belief of the param distribution
        self.theta_priors = self.sim.theta_priors
        "this is for baseline only:"
        self.initial_joint_prob = np.ones((len(self.lambda_list), len(self.theta_list))) / (len(self.lambda_list) * len(self.theta_list)) #do this here to increase speed
        self.traj_h = []
        self.traj_m = []

        "getting true intents of self"
        self.true_intents = []
        for i, par_i in enumerate(self.sim.env.car_par):
            self.true_intents.append(par_i["par"])
        self.true_params = self.sim.true_params
        self.belief_params = self.sim.belief_params
        self.action_set = self.sim.action_set

    # @staticmethod
    def no_inference(self, agents, sim):
        self.frame = self.sim.frame
        print("frame {}".format(sim.frame))
        last_action_h = sim.agents[0].action[self.frame]
        last_action_m = sim.agents[1].action[self.frame]
        return {'last_actions': [last_action_h, last_action_m]}

    def bvp_inference(self, agent, sim):
        """
        INFERENCE MODEL FOR AGENT 1
        Using Q values from value network trained from BVP solver
        When QH also depends on xM,uM
        :return:P(beta_h, beta_m_hat | D(k))
        """

        # ----------------------------#
        # variables:
        # predicted_intent_other: BH hat,
        # predicted_intent_self: BM tilde,
        # predicted_policy_other: QH hat,
        # predicted_policy_self: QM tilde
        # ----------------------------#

        "importing agents information from Autonomous Vehicle (sim.agents)"
        self.frame = self.sim.frame
        self.time = self.sim.time
        assert len(sim.agents[0].state) == len(sim.agents[0].action)
        curr_state_h = sim.agents[0].state[self.frame]
        curr_state_m = sim.agents[1].state[self.frame]

        last_action_h = sim.agents[0].action[self.frame]
        last_state_h = sim.agents[0].state[self.frame - 1]
        last_action_m = sim.agents[1].action[self.frame]
        last_state_m = sim.agents[1].state[self.frame - 1]
        self.traj_h.append([last_state_h, last_action_h])
        self.traj_m.append([last_state_m, last_action_m])

        def action_prob(state_h, state_m, beta_h, beta_m):
            # bvp_action prob is in use instead of this, for faster speed
            """
            calculate action prob for both agents
            :param state_h:
            :param state_m:
            :return: [p_action_H, p_action_M], where p_action = [p_a1, ..., p_a5]
            """

            theta_h, lambda_h = beta_h
            theta_m, lambda_m = beta_m
            action_set = self.action_set
            p1_state_nn = np.array([[state_h[1]], [state_h[3]], [state_m[0]], [state_m[2]]])
            p2_state_nn = np.array([[state_m[0]], [state_m[2]], [state_h[1]], [state_h[3]]])
            _lambda = [lambda_h, lambda_m]

            _p_action_1 = np.zeros(((len(action_set)), len(action_set)))
            _p_action_2 = np.zeros(((len(action_set)), len(action_set)))
            time = np.array([[self.time]])
            dt = self.sim.dt

            for i, p_a_h in enumerate(_p_action_1):
                for j, p_a_m in enumerate(_p_action_1[i]):
                    new_p2_s = dynamics.bvp_dynamics_1d(state_m, action_set[j], dt)
                    new_p1_s = dynamics.bvp_dynamics_1d(state_h, action_set[i], dt)
                    if (theta_h, theta_m) == (1, 5):  # Flip A_AN to NA_A
                        new_p2_state_nn = np.array([[new_p2_s[0]], [new_p2_s[2]], [new_p1_s[1]], [new_p1_s[3]]])
                        q2, q1 = get_Hamilton(new_p2_state_nn, time, np.array([[action_set[j]], [action_set[i]]]),
                                             (theta_m, theta_h))  # NA_A
                    else:  # for A_A, NA_NA, NA_A
                        new_p1_state_nn = np.array([[new_p1_s[1]], [new_p1_s[3]], [new_p2_s[0]], [new_p2_s[2]]])
                        q1, q2 = get_Hamilton(new_p1_state_nn, time, np.array([[action_set[i]], [action_set[j]]]),
                                             (theta_h, theta_m))
                    lamb_Q1 = q1 * lambda_h
                    _p_action_1[i][j] = lamb_Q1
                    lamb_Q2 = q2 * lambda_m
                    _p_action_2[i][j] = lamb_Q2

            "using logsumexp to prevent nan"
            Q1_logsumexp = logsumexp(_p_action_1)
            Q2_logsumexp = logsumexp(_p_action_2)
            "normalizing"
            _p_action_1 -= Q1_logsumexp
            _p_action_2 -= Q2_logsumexp
            _p_action_1 = np.exp(_p_action_1)
            _p_action_2 = np.exp(_p_action_2)
            assert round(np.sum(_p_action_1)) == 1
            assert round(np.sum(_p_action_2)) == 1
            'p action based on observed action of other agent'
            _last_action_h = self.traj_h[-1][1]
            _last_action_m = self.traj_m[-1][1]
            ah = action_set.index(_last_action_h)
            am = action_set.index(_last_action_m)
            _pa_1_t = np.transpose(_p_action_1)
            _pa_1 = _pa_1_t[am]  # column of pa
            _pa_2 = _p_action_2[ah]
            _pa_1 /= np.sum(_pa_1)
            _pa_2 /= np.sum(_pa_2)
            assert round(np.sum(_pa_1)) == 1
            assert round(np.sum(_pa_2)) == 1

            return [_pa_1, _pa_2]  # [exp_Q_h[past_am], exp_Q_m[past_ah]]

        def bvp_action_prob_2(id, p1_state, p2_state, beta_1, beta_2, last_action, time):
            """
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

            dt = self.sim.dt
            time = np.array([[3 - time]])
            "Need state for agent H: xH, vH, xM, vM"
            "Check last_state_h and last_state_m, it should consider to compute the current state"
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

            print("action prob from bvp:", _p_action)
            assert round(np.sum(_p_action)) == 1

            return _p_action  # exp_Q

        def resample(priors, epsilon):
            """
            Resamples the belief P(k-1) from initial belief P0 with some probability of epsilon.
            :return: resampled belief P(k-1) on lambda and theta
            """
            initial_belief = self.p_betas_prior
            resampled_priors = (1 - epsilon) * priors + epsilon * initial_belief
            return resampled_priors

        def prob_beta_pair(prior, traj_h, traj_m):
            """
            MAIN BELIEF UPDATE ALGORITHM
            Calculates probability of beta pair (BH, BM_hat) given past observation D(k).
            :return: P(beta_H, beta_M | D(k)), 8x8
            """

            p_betas_d = np.zeros((len(self.beta_set), len(self.beta_set)))
            assert len(p_betas_d) == len(prior)
            beta_set = self.beta_set
            "Calculate prob of beta pair given D(k)"
            past_state_h, last_action_h = traj_h[-1]
            past_state_m, last_action_m = traj_m[-1]
            last_actions = [last_action_h, last_action_m]
            ah = self.action_set.index(last_action_h)
            am = self.action_set.index(last_action_m)
            ai = [ah, am]
            for i in range(len(p_betas_d)):
                for j in range(len(p_betas_d[i])):
                    p_a_past2 = []
                    for id in range(self.sim.n_agents):
                        p_a_i = bvp_action_prob_2(id, past_state_h, past_state_m,
                                                  beta_set[i], beta_set[j],
                                                  last_actions[id-1], self.time - self.dt)  # should not be self.time - dt
                        p_a_past2.append(p_a_i[ai[id]])
                    "for confirming the two algorithm converges to same value"
                    p_a_pair = p_a_past2[0] * p_a_past2[1]
                    "P(Q2|D(k)) = P(uH, uM|x(k), QH, QM) * P(Q2|D(k-1)) / sum(~)"
                    p_betas_d[i][j] = p_a_pair * prior[i][j]
            p_betas_d /= np.sum(p_betas_d)
            p_betas_d = resample(p_betas_d, epsilon=0.05)
            assert round(np.sum(p_betas_d)) == 1  # make sure this calculation is correct; no need to normalize
            return p_betas_d
            
        def marginal_joint_intent(id, _p_beta_d):
            """
            Get the marginal P(Beta_i|D(k)) from P(beta_H, beta_M|D(k))
            :param id:
            :param _p_beta_d:
            :return: len(lambda_list) x len(theta_list) matrix, where row is the lambda and col is the theta, for visualization purposes
            """
            marginal = []
            for t in self.theta_list:
                marginal.append([])
            "create a 2D array of (lambda, theta) pairs distribution like single agent case"
            half = round(len(self.beta_set) / 2)
            if id == 0:  # H agent
                for i, row in enumerate(_p_beta_d):  # get sum of row
                    if i < half:  # in 1D self.beta, first half are NA, or theta1
                        marginal[0].append(sum(row))
                    else:
                        marginal[1].append(sum(row))
            else:
                for i, col in enumerate(zip(*_p_beta_d)):
                    if i < half:
                        marginal[0].append(sum(col))
                    else:
                        marginal[1].append(sum(col))
            # i-4 if i>3
            id_lambda = marginal.index(max(marginal))
            _best_lambda = self.lambda_list[id_lambda] if id_lambda < half else self.lambda_list[id_lambda - half]
            marginal = np.array(marginal)
            marginal = marginal.transpose()  # Lambdas x Thetas
            return marginal, _best_lambda

        "---------------------------------------------------"
        "calling functions: P(Q2|D), P(beta2|D), P(x(k+1)|D)"
        "---------------------------------------------------"
        'intent and rationality inference'
        if not self.frame == 0:  # update belief only when t =/= 0
            p_beta_d = prob_beta_pair(prior=self.p_betas_prior, traj_h=self.traj_h, traj_m=self.traj_m)
            'recording prior for the next step'
            self.p_betas_prior = p_beta_d
        else:
            p_beta_d = self.beta_initial_belief

        "getting marginal prob for beta_h or beta_m: THIS IS ONLY FOR PLOTTING, NOT DECISION (see more in sim_draw)"
        # for estimating distribution
        p_beta_d_h, best_lambda_h = marginal_joint_intent(id=0, _p_beta_d=p_beta_d)
        p_beta_d_m, best_lambda_m = marginal_joint_intent(id=1, _p_beta_d=p_beta_d)

        "getting most likely action for analysis purpose"
        # # these are only for estimation
        # if self.sim.decision_type_m == 'bvp_empathetic':
        #     p_actions = action_prob(curr_state_h, curr_state_m, new_beta_h, new_beta_m)  # for testing with decision
        # elif self.sim.decision_type_m == 'bvp_non_empathetic':
        #     p_action_1, p_action_2_n = bvp_action_prob_2(curr_state_h, curr_state_m, self.true_params[0], new_beta_m)
        #     p_action_1_n, p_action_2 = action_prob(curr_state_h, curr_state_m, new_beta_h,
        #                                            self.true_params[1])  # for testing with decision
        # for i, p_a in enumerate(p_actions):
        #     # p_a_id = np.unravel_index(p_a.argmax(), p_a.shape)  # choose action for self based on the NE
        #     p_a_id = np.argmax(p_a)
        #     predicted_actions.append(self.action_set[p_a_id])
        last_action_set = [last_action_h, last_action_m]
        a_id = [self.action_set.index(last_action_h), self.action_set.index(last_action_m)]  # index of actions_t-1
        predicted_actions = []
        for id in range(self.sim.n_agents):
            if self.sim.decision_type_m == 'bvp_empathetic' and self.sim.decision_type_h == 'bvp_empathetic':
                'getting best predicted betas for empathetic decision'
                beta_pair_id = np.unravel_index(p_beta_d.argmax(), p_beta_d.shape)
                new_beta_1 = self.beta_set[beta_pair_id[0]]
                new_beta_2 = self.beta_set[beta_pair_id[1]]
                self.past_beta.append([new_beta_1, new_beta_2])

                "getting action probability"
                if id == 0:
                    self.policy_choice[0].append(beta_pair_id[1])
                    p_a_i = bvp_action_prob_2(id, curr_state_h, curr_state_m,
                                              self.true_params[0], new_beta_2, last_action_set[id - 1], self.time)
                elif id == 1:
                    self.policy_choice[1].append(beta_pair_id[0])
                    p_a_i = bvp_action_prob_2(id, curr_state_h, curr_state_m,
                                              new_beta_1, self.true_params[1], last_action_set[id - 1], self.time)

            elif self.sim.decision_type_m == 'bvp_non_empathetic' \
                    and self.sim.decision_type_h == 'bvp_non_empathetic':  # action and beta prediction for NE case
                true_id = self.sim.true_params_id[id]  # true beta of other agent
                if id == 0:  # agent 1's guess of agent 2's beta
                    p_beta_d_i = p_beta_d[true_id]  # get row based on P1's true beta
                    pred_beta_2_id = np.argmax(p_beta_d_i)

                    new_beta_2 = self.beta_set[pred_beta_2_id]

                    self.past_beta.append([new_beta_2])
                    self.policy_choice[0].append(pred_beta_2_id)
                    p_a_i = bvp_action_prob_2(id, curr_state_h, curr_state_m,
                                              self.true_params[0], new_beta_2, last_action_set[id - 1], self.time)
                elif id == 1:  # agent 2's guess at agent 1's beta
                    p_beta_d_i = np.transpose(p_beta_d)[true_id]  # get row based on P1's true beta
                    pred_beta_1_id = np.argmax(p_beta_d_i)
                    new_beta_1 = self.beta_set[pred_beta_1_id]  # new predicted beta for agent 2

                    beta_pair_id = [pred_beta_1_id, pred_beta_2_id]
                    self.past_beta[-1].append(new_beta_1)
                    self.policy_choice[1].append(pred_beta_1_id)
                    p_a_i = bvp_action_prob_2(id, curr_state_h, curr_state_m,
                                              new_beta_1, self.true_params[1], last_action_set[id - 1], self.time)

            # "FOR SINGLE AGENT THAT IS DOING Inference"
            elif self.sim.decision_type_h == 'bvp_empathetic':
                beta_pair_id = np.unravel_index(p_beta_d.argmax(), p_beta_d.shape)
                new_beta_1 = self.beta_set[beta_pair_id[0]]
                new_beta_2 = self.beta_set[beta_pair_id[1]]
                self.past_beta.append([new_beta_1, new_beta_2])

                "getting action probability"
                if id == 0:
                    self.policy_choice[0].append(beta_pair_id[1])
                    p_a_i = bvp_action_prob_2(id, curr_state_h, curr_state_m,
                                              self.true_params[0], new_beta_2, last_action_set[id - 1], self.time)
                elif id == 1:
                    self.policy_choice[1].append(beta_pair_id[0])
                    p_a_i = bvp_action_prob_2(id, curr_state_h, curr_state_m,
                                              new_beta_1, self.true_params[1], last_action_set[id - 1], self.time)

            elif self.sim.decision_type_h == 'bvp_non_empathetic':
                true_id = self.sim.true_params_id[id]  # true beta of other agent
                if id == 0:  # agent 1's guess of agent 2's beta
                    p_beta_d_i = p_beta_d[true_id]  # get row based on P1's true beta
                    pred_beta_2_id = np.argmax(p_beta_d_i)

                    new_beta_2 = self.beta_set[pred_beta_2_id]

                    self.past_beta.append([new_beta_2])
                    self.policy_choice[0].append(pred_beta_2_id)
                    p_a_i = bvp_action_prob_2(id, curr_state_h, curr_state_m,
                                              self.true_params[0], new_beta_2, last_action_set[id - 1], self.time)
                elif id == 1:  # agent 2's guess at agent 1's beta
                    p_beta_d_i = np.transpose(p_beta_d)[true_id]  # get row based on P1's true beta
                    pred_beta_1_id = np.argmax(p_beta_d_i)
                    new_beta_1 = self.beta_set[pred_beta_1_id]  # new predicted beta for agent 2
                    beta_pair_id = [pred_beta_1_id, pred_beta_2_id]
                    self.past_beta[-1].append(new_beta_1)
                    self.policy_choice[1].append(pred_beta_1_id)
                    p_a_i = bvp_action_prob_2(id, curr_state_h, curr_state_m,
                                              new_beta_1, self.true_params[1], last_action_set[id - 1], self.time)

            else:
                print("WARNING! INCORRECT AGENT TYPE FOR BVP INFERENCE")
            best_action_i = self.action_set[np.argmax(p_a_i)]
            predicted_actions.append(best_action_i)

        "Counting chosen parameter in the belief table"
        self.belief_count[beta_pair_id[0]][beta_pair_id[1]] += 1

        # IMPORTANT: Best beta pair =/= Best beta !!!
        # p_theta_prime, suited_lambdas <- predicted_intent other
        # p_betas: [BH x BM]
        # print("state list and prob for H: ", state_list, marginal_state)
        # print("size of state list at t=1", len(state_list[0]))  # should be 5x5 2D
        # variables:
        # predicted_intent_other: BH hat,
        # predicted_intent_self: BM tilde,
        # predicted_policy_other: QH hat,
        # predicted_policy_self: QM tilde
        # print("-Intent_inf- marginal state H: ", marginal_state_h)
        return {'predicted_actions_other': predicted_actions[1],
                'predicted_intent_other': [p_beta_d_m, new_beta_2],
                # 'predicted_states_self': (marginal_state_m, get_state_list(curr_state_m, self.T, self.dt)),
                'predicted_actions_self': predicted_actions[0],
                'predicted_intent_self': [p_beta_d_h, new_beta_1],
                'predicted_intent_all': [p_beta_d, [new_beta_1, new_beta_2]],
                'belief_count': self.belief_count,
                'policy_choice': self.policy_choice,
                'last_actions': last_action_set}

    def bvp_opti_inference(self, agent, sim):
        """
        INFERENCE MODEL FOR AGENT 1
        Using Q values from value network trained from BVP solver
        Continuous version: where action is a continuous set
        :return:P(beta_h, beta_m_hat | D(k))
        """

        # ----------------------------#
        # variables:
        # predicted_intent_other: BH hat,
        # predicted_intent_self: BM tilde,
        # predicted_policy_other: QH hat,
        # predicted_policy_self: QM tilde
        # ----------------------------#

        "importing agents information from Autonomous Vehicle (sim.agents)"
        self.frame = self.sim.frame
        self.time = self.sim.time
        assert len(sim.agents[0].state) == len(sim.agents[0].action)
        curr_state_h = sim.agents[0].state[self.frame]
        curr_state_m = sim.agents[1].state[self.frame]

        last_action_h = sim.agents[0].action[self.frame]
        last_state_h = sim.agents[0].state[self.frame - 1]
        last_action_m = sim.agents[1].action[self.frame]
        last_state_m = sim.agents[1].state[self.frame - 1]
        self.traj_h.append([last_state_h, last_action_h])
        self.traj_m.append([last_state_m, last_action_m])

        def resample(priors, epsilon):
            """
            Resamples the belief P(k-1) from initial belief P0 with some probability of epsilon.
            :return: resampled belief P(k-1) on lambda and theta
            """
            initial_belief = self.p_betas_prior
            resampled_priors = (1 - epsilon) * priors + epsilon * initial_belief
            return resampled_priors

        def q_func_integrate_helper1(Ui, X, T, last_u_other, theta, lambda_1, deltaT):
            last_action = 0
            q1, q2 = get_Hamilton(X, T, np.array([[Ui], [last_u_other]]), theta)
            exp_q1 = np.exp(lambda_1 * q1[0][0])
            assert exp_q1 != 0

            return exp_q1

        def integrate_action_1(X, last_u_other, T, theta, lambda_1, deltaT):  # TODO: HOW TO INTEGRATE exp(Q)??
            """
            Integrates for the denominator of action prob: exp(lambda Q)
            :param id: which agent
            :param X:
            :param T: current time
            :param theta:
            :param deltaT:
            :return:
            """

            return quad(q_func_integrate_helper1, -5, 10, args=(X, T, last_u_other, theta, lambda_1, deltaT))[0]  # action bounds: -5 to 10

        def q_func_integrate_helper2(Ui, X, T, last_u_other, theta, lambda_2, deltaT):
            last_action = 0
            q1, q2 = get_Hamilton(X, T, np.array([[last_u_other], [Ui]]), theta)
            exp_q2 = np.exp(lambda_2 * q2[0][0])
            print(q2, exp_q2)
            assert exp_q2 != 0

            return exp_q2

        def integrate_action_2(X, last_u_other, T, theta, lambda_2, deltaT):  # TODO: HOW TO INTEGRATE exp(Q)??
            """
            Integrates for the denominator of action prob: exp(lambda Q)
            :param id: which agent
            :param X:
            :param T: current time
            :param theta:
            :param deltaT:
            :return:
            """

            return quad(q_func_integrate_helper2, -5, 10, args=(X, T, last_u_other, theta, lambda_2, deltaT))[0]  # action bounds: -5 to 10

        def action_prob_continuous(id, p1_state, p2_state, beta_1, beta_2, last_action, time):
            """
            Returns probability of the last action observed.
            :param id:
            :param p1_state:
            :param p2_state:
            :param beta_1:
            :param beta_2:
            :param last_action:
            :param time:
            :return:
            """
            dt = self.dt
            # new_p1_s = dynamics.bvp_dynamics_1d(p1_state, last_action[0], dt)
            # new_p2_s = dynamics.bvp_dynamics_1d(p2_state, last_action[1], dt)
            x1_nn = np.array([[p1_state[1]], [p1_state[3]], [p2_state[0]], [p2_state[2]]])  # using old state with last action
            theta = (beta_1[0], beta_2[0])
            lambdas = (beta_1[1], beta_2[1])
            last_actions_nn = np.array([[last_action[0]], [last_action[1]]])
            time = np.array([[time]])
            q1, q2 = get_Hamilton(x1_nn, time, last_actions_nn, theta)
            print("current state (inference 2564): ", x1_nn)

            if id == 0:
                p_action = np.exp(lambdas[0] * q1) / integrate_action_1(x1_nn, last_action[1], time, theta, lambdas[0],dt)
            elif id == 1:
                p_action = np.exp(lambdas[1] * q2) / integrate_action_2(x1_nn, last_action[0], time, theta, lambdas[1], dt)
            else:
                print("WARNING ID OUT OF RANGE")

            return p_action

        def bvp_optimized_action(id, p1_state, p2_state, beta_1, beta_2, last_action, time):
            """
            Uses scipy optimizer to find best action
            :return:
            """
            "sorting states to obtain action from pre-trained model"
            # y direction only for M, x direction only for HV
            self.frame = self.sim.frame
            self.time = self.sim.time
            dt = self.dt

            # x_nn should be new x after the action
            time = np.array([[self.time]])

            x1_nn = np.array(
                [[p1_state[1]], [p1_state[3]], [p2_state[0]], [p2_state[2]]])  # using old state with last action

            def q_function_helper1(action):  # for agent 1
                q1, q2 = get_Hamilton(x1_nn, time, np.array([[action], [last_action]]),
                                     (beta_1[0], beta_2[0]))
                q1 = -q1[0][0]
                # print(action, q1)
                return q1

            def q_function_helper2(action):  # for agent 2
                q1, q2 = get_Hamilton(x1_nn, time, np.array([[last_action], [action]]),
                                     (beta_1[0], beta_2[0]))
                q2 = -q2[0][0]
                # print(action, q2)
                return q2

            if id == 0:
                res = minimize_scalar(q_function_helper1, bounds=(-5, 10), method="bounded")
            elif id == 1:
                res = minimize_scalar(q_function_helper2, bounds=(-5, 10), method="bounded")
            else:
                print("agent id out of range!!!")
                res = None

            return res

        def beta_prob_continuous(prior, traj_h, traj_m):
            """
            MAIN BELIEF UPDATE ALGORITHM
            Calculates probability of beta pair (BH, BM_hat) given past observation D(k).
            :return: P(beta_H, beta_M | D(k)), 8x8
            """

            p_betas_d = np.zeros((len(self.beta_set), len(self.beta_set)))
            assert len(p_betas_d) == len(prior)
            beta_set = self.beta_set
            "Calculate prob of beta pair given D(k)"
            past_state_h, last_action_h = traj_h[-1]
            past_state_m, last_action_m = traj_m[-1]
            last_actions = [last_action_h, last_action_m]

            for i in range(len(p_betas_d)):
                for j in range(len(p_betas_d[i])):

                    "method 1: get whole action prob table"
                    # p_a_past, p_action = past_action_prob(past_state_h, past_state_m, beta_set[i], beta_set[j],
                    #                                       last_action_h, last_action_m)  # action prob for last step!

                    "method 2: only get 1 row of p_action for faster speed"
                    p_a_past = []
                    for id in range(self.sim.n_agents):
                        p_a_i = action_prob_continuous(id, past_state_h, past_state_m,
                                                  beta_set[i], beta_set[j],
                                                  last_actions, self.time - self.dt)  # last time step
                        p_a_past.append(p_a_i)
                    "for confirming the two algorithm converges to same value"
                    p_a_pair = p_a_past[0] * p_a_past[1]
                    "P(Q2|D(k)) = P(uH, uM|x(k), QH, QM) * P(Q2|D(k-1)) / sum(~)"
                    p_betas_d[i][j] = p_a_pair * prior[i][j]
            p_betas_d /= np.sum(p_betas_d)
            p_betas_d = resample(p_betas_d, epsilon=0.05)
            assert round(np.sum(p_betas_d)) == 1  # make sure this calculation is correct; no need to normalize
            return p_betas_d

        def marginal_joint_intent(id, _p_beta_d):
            """
            Get the marginal P(Beta_i|D(k)) from P(beta_H, beta_M|D(k))
            :param id:
            :param _p_beta_d:
            :return: len(lambda_list) x len(theta_list) matrix, where row is the lambda and col is the theta, for visualization purposes
            """
            marginal = []
            for t in self.theta_list:
                marginal.append([])
            "create a 2D array of (lambda, theta) pairs distribution like single agent case"
            half = round(len(self.beta_set) / 2)
            if id == 0:  # H agent
                for i, row in enumerate(_p_beta_d):  # get sum of row
                    if i < half:  # in 1D self.beta, first half are NA, or theta1
                        marginal[0].append(sum(row))
                    else:
                        marginal[1].append(sum(row))
            else:
                for i, col in enumerate(zip(*_p_beta_d)):
                    if i < half:
                        marginal[0].append(sum(col))
                    else:
                        marginal[1].append(sum(col))
            # i-4 if i>3
            id_lambda = marginal.index(max(marginal))
            _best_lambda = self.lambda_list[id_lambda] if id_lambda < half else self.lambda_list[id_lambda - half]
            marginal = np.array(marginal)
            marginal = marginal.transpose()  # Lambdas x Thetas
            return marginal, _best_lambda

        "---------------------------------------------------"
        "calling functions: P(Q2|D), P(beta2|D), P(x(k+1)|D)"
        "---------------------------------------------------"
        "get last predicted beta pair"
        'intent and rationality inference'
        # using traj[-1] to calculate p_action
        if not self.frame == 0:  # update belief only when t =/= 0
            p_beta_d = beta_prob_continuous(prior=self.p_betas_prior, traj_h=self.traj_h, traj_m=self.traj_m)
            'recording prior for the next step'
            self.p_betas_prior = p_beta_d

        else:  # last action is generated in env during startup using given parameters
            p_beta_d = self.beta_initial_belief

        last_action_set = [last_action_h, last_action_m]
        a_id = [self.action_set.index(last_action_h), self.action_set.index(last_action_m)]  # index of actions_t-1
        "getting marginal prob for beta_h or beta_m: THIS IS ONLY FOR PLOTTING, NOT DECISION (see more in sim_draw)"
        # for estimating distribution
        p_beta_d_h, best_lambda_h = marginal_joint_intent(id=0, _p_beta_d=p_beta_d)
        p_beta_d_m, best_lambda_m = marginal_joint_intent(id=1, _p_beta_d=p_beta_d)

        "getting most likely action for analysis purpose"
        predicted_actions = []
        for id in range(self.sim.n_agents):
            if self.sim.decision_type_m == 'bvp_e_optimize' and self.sim.decision_type_h == 'bvp_e_optimize':
                'getting best predicted betas for empathetic decision'
                beta_pair_id = np.unravel_index(p_beta_d.argmax(), p_beta_d.shape)
                new_beta_1 = self.beta_set[beta_pair_id[0]]
                new_beta_2 = self.beta_set[beta_pair_id[1]]
                self.past_beta.append([new_beta_1, new_beta_2])

                "getting action probability"
                if id == 0:
                    self.policy_choice[0].append(beta_pair_id[1])
                    a_i = bvp_optimized_action(id, curr_state_h, curr_state_m,
                                              self.true_params[0], new_beta_2, last_action_set[id - 1], self.time)
                elif id == 1:
                    self.policy_choice[1].append(beta_pair_id[0])
                    a_i = bvp_optimized_action(id, curr_state_h, curr_state_m,
                                              new_beta_1, self.true_params[1], last_action_set[id - 1], self.time)

            elif self.sim.decision_type_m == 'bvp_ne_optimize' \
                    and self.sim.decision_type_h == 'bvp_ne_optimize':  # action and beta prediction for NE case
                true_id = self.sim.true_params_id[id]  # true beta of other agent
                if id == 0:  # agent 1's guess of agent 2's beta
                    p_beta_d_i = p_beta_d[true_id]  # get row based on P1's true beta
                    pred_beta_2_id = np.argmax(p_beta_d_i)

                    new_beta_2 = self.beta_set[pred_beta_2_id]

                    self.past_beta.append([new_beta_2])
                    self.policy_choice[0].append(pred_beta_2_id)
                    a_i = bvp_optimized_action(id, curr_state_h, curr_state_m,
                                              self.true_params[0], new_beta_2, last_action_set[id - 1], self.time)
                elif id == 1:  # agent 2's guess at agent 1's beta
                    p_beta_d_i = np.transpose(p_beta_d)[true_id]  # get row based on P1's true beta
                    pred_beta_1_id = np.argmax(p_beta_d_i)
                    new_beta_1 = self.beta_set[pred_beta_1_id]  # new predicted beta for agent 2
                    beta_pair_id = [pred_beta_1_id, pred_beta_2_id]
                    self.past_beta[-1].append(new_beta_1)
                    self.policy_choice[1].append(pred_beta_1_id)
                    a_i = bvp_optimized_action(id, curr_state_h, curr_state_m,
                                                 new_beta_1, self.true_params[1], last_action_set[id - 1], self.time)

                "FOR SINGLE AGENT THAT IS DOING Inference"
            elif self.sim.decision_type_h == 'bvp_e_optimize':
                beta_pair_id = np.unravel_index(p_beta_d.argmax(), p_beta_d.shape)
                new_beta_1 = self.beta_set[beta_pair_id[0]]
                new_beta_2 = self.beta_set[beta_pair_id[1]]
                self.past_beta.append([new_beta_1, new_beta_2])

                "getting action probability"
                if id == 0:
                    self.policy_choice[0].append(beta_pair_id[1])
                    a_i = bvp_optimized_action(id, curr_state_h, curr_state_m,
                                              self.true_params[0], new_beta_2, last_action_set[id - 1], self.time)
                elif id == 1:
                    self.policy_choice[1].append(beta_pair_id[0])
                    a_i = bvp_optimized_action(id, curr_state_h, curr_state_m,
                                              new_beta_1, self.true_params[1], last_action_set[id - 1], self.time)

            elif self.sim.decision_type_h == 'bvp_ne_optimize':
                true_id = self.sim.true_params_id[id]  # true beta of other agent
                if id == 0:  # agent 1's guess of agent 2's beta
                    p_beta_d_i = p_beta_d[true_id]  # get row based on P1's true beta
                    pred_beta_2_id = np.argmax(p_beta_d_i)

                    new_beta_2 = self.beta_set[pred_beta_2_id]

                    self.past_beta.append([new_beta_2])
                    self.policy_choice[0].append(pred_beta_2_id)
                    a_i = bvp_optimized_action(id, curr_state_h, curr_state_m,
                                              self.true_params[0], new_beta_2, last_action_set[id - 1], self.time)
                elif id == 1:  # agent 2's guess at agent 1's beta
                    p_beta_d_i = np.transpose(p_beta_d)[true_id]  # get row based on P1's true beta
                    pred_beta_1_id = np.argmax(p_beta_d_i)
                    new_beta_1 = self.beta_set[pred_beta_1_id]  # new predicted beta for agent 2
                    beta_pair_id = [pred_beta_1_id, pred_beta_2_id]
                    self.past_beta[-1].append(new_beta_1)
                    self.policy_choice[1].append(pred_beta_1_id)
                    a_i = bvp_optimized_action(id, curr_state_h, curr_state_m,
                                              new_beta_1, self.true_params[1], last_action_set[id - 1], self.time)

            else:
                print("WARNING! INCORRECT AGENT TYPE FOR BVP INFERENCE")
            # best_action_i = self.action_set[np.argmax(p_a_i)]
            predicted_actions.append(a_i.x)

        "Counting chosen parameter in the belief table"
        self.belief_count[beta_pair_id[0]][beta_pair_id[1]] += 1

        # IMPORTANT: Best beta pair =/= Best beta !!!
        return {
            'predicted_actions_other': predicted_actions[1],
            'predicted_intent_other': [p_beta_d_m, new_beta_2],
            'predicted_actions_self': predicted_actions[0],
            'predicted_intent_self': [p_beta_d_h, new_beta_1],
            'predicted_intent_all': [p_beta_d, [new_beta_1, new_beta_2]],
            'belief_count': self.belief_count,
            'policy_choice': self.policy_choice,
            'last_actions': last_action_set}
