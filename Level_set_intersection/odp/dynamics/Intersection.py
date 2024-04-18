"""
Uncontrolled intersection is to compute the two-player general-sum differential with state penalty
V = min{\int_t^T (l+u^2) + g(x(T))}, u is control effort, l is state penalty, g is terminal loss
"""

import heterocl as hcl
import math
import numpy as np

""" INTERSECTION DYNAMICS IMPLEMENTATION 
 d1_dot = v1
 v1_dot = u1
 d2_dot = v2
 v2_dot = u2
 """

class Intersection:
    def __init__(self, x=[0,0,0,0], uMin = [-5,-5], uMax = [10,10], uMode="min", dMode="max"):

        """Creates a Dublin Car with the following states:
           X position, Y position, acceleration, heading

           The first element of user control and disturbance is acceleration
           The second element of user control and disturbance is heading


        Args:
            x (list, optional): Initial state . Defaults to [0,0,0,0].
            uMin (list, optional): Lowerbound of user control. Defaults to [-1,-1].
            uMax (list, optional): Upperbound of user control.
                                   Defaults to [1,1].
            dMin (list, optional): Lowerbound of disturbance to user control, . Defaults to [-0.25,-0.25].
            dMax (list, optional): Upperbound of disturbance to user control. Defaults to [0.25,0.25].
            uMode (str, optional): Accepts either "min" or "max".
                                   * "min" : have optimal control reach goal
                                   * "max" : have optimal control avoid goal
                                   Defaults to "min".
            dMode (str, optional): Accepts whether "min" or "max" and should be opposite of uMode.
                                   Defaults to "max".
        """
        self.x = x
        self.uMax = uMax
        self.uMin = uMin
        assert uMode in ["min", "max"]
        self.uMode = uMode
        if uMode == "min":
            assert dMode == "max"
        else:
            assert dMode == "min"
        self.dMode = dMode

        self.beta = 10000
        self.theta1 = 1
        self.theta2 = 5

        # weight for terminal lose
        self.alpha = 1e-06  # 1e-06

        # Length for each vehicle
        self.L1 = 3
        self.L2 = 3

        # Length for each vehicle
        self.W1 = 1.5
        self.W2 = 1.5

        # Road length setting
        self.R1 = 70
        self.R2 = 70

    def opt_u1(self, t, state, spat_deriv):
        """
        :param t: time t
        :param state: tuple of coordinates
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return:
        """
        # System dynamics
        # d1_dot = v1
        # v1_dot = u1
        # d2_dot = v2
        # v2_dot = u2

        # Graph takes in 4 possible inputs, by default, for now
        opt_u1 = hcl.scalar(0, "opt_u1")
        # Just create and pass back, even though they're not used
        u1in1 = hcl.scalar(0, "u1in1")
        u1in2 = hcl.scalar(0, "u1in2")
        u1in3 = hcl.scalar(0, "u1in3")

        u1_term = hcl.scalar(0, "u1_term")
        u1_term[0] = 0.5*spat_deriv[1]
        opt_u1[0] = u1_term[0]

        with hcl.if_(u1_term[0] > 10):
            opt_u1[0] = 10
        with hcl.elif_(u1_term[0] < -5):
            opt_u1[0] = -5

        # return 1, 2, 3 even if you don't use them
        return (opt_u1[0], u1in1[0], u1in2[0], u1in3[0])

    def opt_u2(self, t, state, spat_deriv):
        """
        :param t: time t
        :param state: tuple of coordinates
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return:
        """
        # System dynamics
        # d1_dot = v1
        # v1_dot = u1
        # d2_dot = v2
        # v2_dot = u2

        # Graph takes in 4 possible inputs, by default, for now
        opt_u2 = hcl.scalar(0, "opt_u2")
        # Just create and pass back, even though they're not used
        u2in1 = hcl.scalar(0, "u2in1")
        u2in2 = hcl.scalar(0, "u2in2")
        u2in3 = hcl.scalar(0, "u2in3")

        u2_term = hcl.scalar(0, "u2_term")
        u2_term[0] = 0.5*spat_deriv[3]
        opt_u2[0] = u2_term[0]

        with hcl.if_(u2_term[0] > 10):
            opt_u2[0] = 10
        with hcl.elif_(u2_term[0] < -5):
            opt_u2[0] = -5

        return (opt_u2[0], u2in1[0], u2in2[0], u2in3[0])

    def dynamics(self, t, state, u1opt, u2opt):

        d1_dot = hcl.scalar(0, "d1_dot")
        v1_dot = hcl.scalar(0, "v1_dot")
        d2_dot = hcl.scalar(0, "d2_dot")
        v2_dot = hcl.scalar(0, "v2_dot")

        d1_dot[0] = state[1]
        v1_dot[0] = u1opt[0]
        d2_dot[0] = state[3]
        v2_dot[0] = u2opt[0]

        return (d1_dot[0], v1_dot[0], d2_dot[0], v2_dot[0])

    def instantaneous_loss(self, t, state, u1opt, u2opt):

        loss1 = hcl.scalar(0, "loss1")
        loss2 = hcl.scalar(0, "loss2")
        l1 = hcl.scalar(0, "l1")
        l2 = hcl.scalar(0, "l2")

        x1in = hcl.scalar(0, "x1in")
        x1out = hcl.scalar(0, "x1out")
        x2in = hcl.scalar(0, "x2in")
        x2out = hcl.scalar(0, "x2out")

        x1in[0] = (state[0] - self.R1 / 2 + self.theta1 * self.W2 / 2) * 5  # 5, 0.1
        x1out[0] = -(state[0] - self.R1 / 2 - self.W2 / 2 - self.L1) * 5
        x2in[0] = (state[2] - self.R2 / 2 + self.theta2 * self.W1 / 2) * 5
        x2out[0] = -(state[2] - self.R2 / 2 - self.W1 / 2 - self.L2) * 5

        l1[0] = self.beta * (1/(1 + hcl.exp(-x1in[0])))*(1/(1 + hcl.exp(-x1out[0])))*(1/(1 + hcl.exp(-x2in[0])))*(1/(1 + hcl.exp(-x2out[0])))
        l2[0] = self.beta * (1/(1 + hcl.exp(-x1in[0])))*(1/(1 + hcl.exp(-x1out[0])))*(1/(1 + hcl.exp(-x2in[0])))*(1/(1 + hcl.exp(-x2out[0])))

        loss1[0] = l1[0] + u1opt[0]*u1opt[0]
        loss2[0] = l2[0] + u2opt[0]*u2opt[0]

        """
        when we compute value function without instantaneous loss, we just need to compute the instantaneous loss using
        the level set method and V_{continuous} = V_{discontinuous} - instantaneous loss
        which means the output is (l1[0], l2[0]) rather than (loss1[0], loss2[0])
        """

        return (loss1[0], loss2[0])

    # The below function can have whatever form or parameters users want
    # These functions are not used in HeteroCL program, hence is pure Python code and
    # can be used after the value function has been obtained.

    def optCtrl_inPython(self, spat_deriv1, spat_deriv2):
        """
        :param t: time t
        :param state: tuple of coordinates
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return:
        """
        # System dynamics
        # d1_dot = v1
        # v1_dot = u1
        # d2_dot = v2
        # v2_dot = u2

        opt_u1 = np.array([0.5*spat_deriv1[1]])
        opt_u2 = np.array([0.5*spat_deriv2[3]])

        max_acc = 10
        min_acc = -5
        opt_u1[np.where(opt_u1 > max_acc)] = max_acc
        opt_u1[np.where(opt_u1 < min_acc)] = min_acc
        opt_u2[np.where(opt_u2 > max_acc)] = max_acc
        opt_u2[np.where(opt_u2 < min_acc)] = min_acc

        return opt_u1, opt_u2



