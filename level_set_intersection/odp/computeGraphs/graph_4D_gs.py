import heterocl as hcl
import numpy as np
from odp.computeGraphs.CustomGraphFunctions import *
from odp.spatialDerivatives.first_orderENO4D import *
from odp.spatialDerivatives.second_orderENO4D import *

########################## 4D Graph definition #################################
def graph_4D_gs(my_object, g, accuracy, generate_SpatDeriv=False, deriv_dim=1):
    V1_f = hcl.placeholder(tuple(g.pts_each_dim), name="V1_f", dtype=hcl.Float())
    V1_init = hcl.placeholder(tuple(g.pts_each_dim), name="V1_init", dtype=hcl.Float())
    V2_f = hcl.placeholder(tuple(g.pts_each_dim), name="V2_f", dtype=hcl.Float())
    V2_init = hcl.placeholder(tuple(g.pts_each_dim), name="V2_init", dtype=hcl.Float())
    t = hcl.placeholder((2,), name="t", dtype=hcl.Float())
    probe = hcl.placeholder(tuple(g.pts_each_dim), name="probe", dtype=hcl.Float())

    # Positions vector
    x1 = hcl.placeholder((g.pts_each_dim[0],), name="x1", dtype=hcl.Float())
    x2 = hcl.placeholder((g.pts_each_dim[1],), name="x2", dtype=hcl.Float())
    x3 = hcl.placeholder((g.pts_each_dim[2],), name="x3", dtype=hcl.Float())
    x4 = hcl.placeholder((g.pts_each_dim[3],), name="x4", dtype=hcl.Float())

    def graph_create(V1_new, V1_init, V2_new, V2_init, x1, x2, x3, x4, t, probe):
        # Specify intermediate tensors
        deriv1_diff1 = hcl.compute(V1_init.shape, lambda *x: 0, "deriv1_diff1")
        deriv1_diff2 = hcl.compute(V1_init.shape, lambda *x: 0, "deriv1_diff2")
        deriv1_diff3 = hcl.compute(V1_init.shape, lambda *x: 0, "deriv1_diff3")
        deriv1_diff4 = hcl.compute(V1_init.shape, lambda *x: 0, "deriv1_diff4")
        deriv2_diff1 = hcl.compute(V2_init.shape, lambda *x: 0, "deriv2_diff1")
        deriv2_diff2 = hcl.compute(V2_init.shape, lambda *x: 0, "deriv2_diff2")
        deriv2_diff3 = hcl.compute(V2_init.shape, lambda *x: 0, "deriv2_diff3")
        deriv2_diff4 = hcl.compute(V2_init.shape, lambda *x: 0, "deriv2_diff4")

        # Maximum derivative for each dim
        max1_deriv1 = hcl.scalar(-1e9, "max1_deriv1")
        max1_deriv2 = hcl.scalar(-1e9, "max1_deriv2")
        max1_deriv3 = hcl.scalar(-1e9, "max1_deriv3")
        max1_deriv4 = hcl.scalar(-1e9, "max1_deriv4")
        max2_deriv1 = hcl.scalar(-1e9, "max2_deriv1")
        max2_deriv2 = hcl.scalar(-1e9, "max2_deriv2")
        max2_deriv3 = hcl.scalar(-1e9, "max2_deriv3")
        max2_deriv4 = hcl.scalar(-1e9, "max2_deriv4")

        # Min derivative for each dim
        min1_deriv1 = hcl.scalar(1e9, "min1_deriv1")
        min1_deriv2 = hcl.scalar(1e9, "min1_deriv2")
        min1_deriv3 = hcl.scalar(1e9, "min1_deriv3")
        min1_deriv4 = hcl.scalar(1e9, "min1_deriv4")
        min2_deriv1 = hcl.scalar(1e9, "min2_deriv1")
        min2_deriv2 = hcl.scalar(1e9, "min2_deriv2")
        min2_deriv3 = hcl.scalar(1e9, "min2_deriv3")
        min2_deriv4 = hcl.scalar(1e9, "min2_deriv4")

        # These variables are used to dissipation calculation
        max_alpha1 = hcl.scalar(-1e9, "max_alpha1")
        max_alpha2 = hcl.scalar(-1e9, "max_alpha2")
        max_alpha3 = hcl.scalar(-1e9, "max_alpha3")
        max_alpha4 = hcl.scalar(-1e9, "max_alpha4")

        def step_bound():  # Function to calculate time step
            stepBoundInv = hcl.scalar(0, "stepBoundInv")
            stepBound = hcl.scalar(0, "stepBound")
            stepBoundInv[0] = max_alpha1[0] / g.dx[0] + max_alpha2[0] / g.dx[1] + max_alpha3[0] / g.dx[2] + max_alpha4[0] / g.dx[3]

            stepBound[0] = 0.8 / stepBoundInv[0]
            with hcl.if_(stepBound > t[1] - t[0]):
                stepBound[0] = t[1] - t[0]

            # Update the lower time ranges
            t[0] = t[0] + stepBound[0]
            return stepBound[0]

        # Calculate Hamiltonian for every grid point in V_init
        with hcl.Stage("Hamiltonian"):
            with hcl.for_(0, V1_init.shape[0], name="i") as i:
                with hcl.for_(0, V1_init.shape[1], name="j") as j:
                    with hcl.for_(0, V1_init.shape[2], name="k") as k:
                        with hcl.for_(0, V1_init.shape[3], name="l") as l:
                            # Variables to calculate dV_dx
                            dV1_dx1_L = hcl.scalar(0, "dV1_dx1_L")
                            dV1_dx1_R = hcl.scalar(0, "dV1_dx1_R")
                            dV1_dx1 = hcl.scalar(0, "dV1_dx1")
                            dV1_dx2_L = hcl.scalar(0, "dV1_dx2_L")
                            dV1_dx2_R = hcl.scalar(0, "dV1_dx2_R")
                            dV1_dx2 = hcl.scalar(0, "dV1_dx2")
                            dV1_dx3_L = hcl.scalar(0, "dV1_dx3_L")
                            dV1_dx3_R = hcl.scalar(0, "dV1_dx3_R")
                            dV1_dx3 = hcl.scalar(0, "dV1_dx3")
                            dV1_dx4_L = hcl.scalar(0, "dV1_dx4_L")
                            dV1_dx4_R = hcl.scalar(0, "dV1_dx4_R")
                            dV1_dx4 = hcl.scalar(0, "dV1_dx4")

                            dV2_dx1_L = hcl.scalar(0, "dV2_dx1_L")
                            dV2_dx1_R = hcl.scalar(0, "dV2_dx1_R")
                            dV2_dx1 = hcl.scalar(0, "dV2_dx1")
                            dV2_dx2_L = hcl.scalar(0, "dV2_dx2_L")
                            dV2_dx2_R = hcl.scalar(0, "dV2_dx2_R")
                            dV2_dx2 = hcl.scalar(0, "dV2_dx2")
                            dV2_dx3_L = hcl.scalar(0, "dV2_dx3_L")
                            dV2_dx3_R = hcl.scalar(0, "dV2_dx3_R")
                            dV2_dx3 = hcl.scalar(0, "dV2_dx3")
                            dV2_dx4_L = hcl.scalar(0, "dV2_dx4_L")
                            dV2_dx4_R = hcl.scalar(0, "dV2_dx4_R")
                            dV2_dx4 = hcl.scalar(0, "dV2_dx4")

                            # No tensor slice operation
                            # dV_dx_L[0], dV_dx_R[0] = spa_derivX(i, j, k)
                            if accuracy == "low":
                                dV1_dx1_L[0], dV1_dx1_R[0] = spa_derivX1_4d(i, j, k, l, V1_init, g)
                                dV1_dx2_L[0], dV1_dx2_R[0] = spa_derivX2_4d(i, j, k, l, V1_init, g)
                                dV1_dx3_L[0], dV1_dx3_R[0] = spa_derivX3_4d(i, j, k, l, V1_init, g)
                                dV1_dx4_L[0], dV1_dx4_R[0] = spa_derivX4_4d(i, j, k, l, V1_init, g)
                                dV2_dx1_L[0], dV2_dx1_R[0] = spa_derivX1_4d(i, j, k, l, V2_init, g)
                                dV2_dx2_L[0], dV2_dx2_R[0] = spa_derivX2_4d(i, j, k, l, V2_init, g)
                                dV2_dx3_L[0], dV2_dx3_R[0] = spa_derivX3_4d(i, j, k, l, V2_init, g)
                                dV2_dx4_L[0], dV2_dx4_R[0] = spa_derivX4_4d(i, j, k, l, V2_init, g)
                            if accuracy == "medium":
                                dV1_dx1_L[0], dV1_dx1_R[0] = secondOrderX1_4d(i, j, k, l, V1_init, g)
                                dV1_dx2_L[0], dV1_dx2_R[0] = secondOrderX2_4d(i, j, k, l, V1_init, g)
                                dV1_dx3_L[0], dV1_dx3_R[0] = secondOrderX3_4d(i, j, k, l, V1_init, g)
                                dV1_dx4_L[0], dV1_dx4_R[0] = secondOrderX4_4d(i, j, k, l, V1_init, g)
                                dV2_dx1_L[0], dV2_dx1_R[0] = secondOrderX1_4d(i, j, k, l, V2_init, g)
                                dV2_dx2_L[0], dV2_dx2_R[0] = secondOrderX2_4d(i, j, k, l, V2_init, g)
                                dV2_dx3_L[0], dV2_dx3_R[0] = secondOrderX3_4d(i, j, k, l, V2_init, g)
                                dV2_dx4_L[0], dV2_dx4_R[0] = secondOrderX4_4d(i, j, k, l, V2_init, g)

                            # Saves spatial derivative diff into tables
                            deriv1_diff1[i, j, k, l] = dV1_dx1_R[0] - dV1_dx1_L[0]
                            deriv1_diff2[i, j, k, l] = dV1_dx2_R[0] - dV1_dx2_L[0]
                            deriv1_diff3[i, j, k, l] = dV1_dx3_R[0] - dV1_dx3_L[0]
                            deriv1_diff4[i, j, k, l] = dV1_dx4_R[0] - dV1_dx4_L[0]

                            deriv2_diff1[i, j, k, l] = dV2_dx1_R[0] - dV2_dx1_L[0]
                            deriv2_diff2[i, j, k, l] = dV2_dx2_R[0] - dV2_dx2_L[0]
                            deriv2_diff3[i, j, k, l] = dV2_dx3_R[0] - dV2_dx3_L[0]
                            deriv2_diff4[i, j, k, l] = dV2_dx4_R[0] - dV2_dx4_L[0]

                            # Calculate average gradient
                            dV1_dx1[0] = (dV1_dx1_L + dV1_dx1_R) / 2
                            dV1_dx2[0] = (dV1_dx2_L + dV1_dx2_R) / 2
                            dV1_dx3[0] = (dV1_dx3_L + dV1_dx3_R) / 2
                            dV1_dx4[0] = (dV1_dx4_L + dV1_dx4_R) / 2

                            dV2_dx1[0] = (dV2_dx1_L + dV2_dx1_R) / 2
                            dV2_dx2[0] = (dV2_dx2_L + dV2_dx2_R) / 2
                            dV2_dx3[0] = (dV2_dx3_L + dV2_dx3_R) / 2
                            dV2_dx4[0] = (dV2_dx4_L + dV2_dx4_R) / 2

                            #probe[i,j,k,l] = dV_dx2[0]
                            # Find optimal control u1
                            u1opt = my_object.opt_u1(t, (x1[i], x2[j], x3[k], x4[l]),
                                                    (dV1_dx1[0], dV1_dx2[0], dV1_dx3[0], dV1_dx4[0]))

                            # Find optimal control u2
                            u2opt = my_object.opt_u2(t, (x1[i], x2[j], x3[k], x4[l]),
                                                    (dV2_dx1[0], dV2_dx2[0], dV2_dx3[0], dV2_dx4[0]))

                            # Find rates of changes based on dynamics equation
                            dx1_dt, dx2_dt, dx3_dt, dx4_dt = my_object.dynamics(t, (x1[i], x2[j], x3[k], x4[l]), u1opt, u2opt)

                            loss1, loss2 = my_object.instantaneous_loss(t, (x1[i], x2[j], x3[k], x4[l]), u1opt, u2opt)

                            # Calculate Hamiltonian terms:
                            V1_new[i, j, k, l] = -(dx1_dt * dV1_dx1[0] + dx2_dt * dV1_dx2[0] + dx3_dt * dV1_dx3[0] + dx4_dt * dV1_dx4[0] - loss1)
                            V2_new[i, j, k, l] = -(dx1_dt * dV2_dx1[0] + dx2_dt * dV2_dx2[0] + dx3_dt * dV2_dx3[0] + dx4_dt * dV2_dx4[0] - loss2)
                            # Debugging
                            # V_new[i, j, k, l] = dV_dx2[0]
                            probe[i, j, k, l] = V1_init[i, j, k, l]

                            # Get derivMin for player 1
                            with hcl.if_(dV1_dx1_L[0] < min1_deriv1[0]):
                                min1_deriv1[0] = dV1_dx1_L[0]
                            with hcl.if_(dV1_dx1_R[0] < min1_deriv1[0]):
                                min1_deriv1[0] = dV1_dx1_R[0]

                            with hcl.if_(dV1_dx2_L[0] < min1_deriv2[0]):
                                min1_deriv2[0] = dV1_dx2_L[0]
                            with hcl.if_(dV1_dx2_R[0] < min1_deriv2[0]):
                                min1_deriv2[0] = dV1_dx2_R[0]

                            with hcl.if_(dV1_dx3_L[0] < min1_deriv3[0]):
                                min1_deriv3[0] = dV1_dx3_L[0]
                            with hcl.if_(dV1_dx3_R[0] < min1_deriv3[0]):
                                min1_deriv3[0] = dV1_dx3_R[0]

                            with hcl.if_(dV1_dx4_L[0] < min1_deriv4[0]):
                                min1_deriv4[0] = dV1_dx4_L[0]
                            with hcl.if_(dV1_dx4_R[0] < min1_deriv4[0]):
                                min1_deriv4[0] = dV1_dx4_R[0]

                            # Get derivMax for player 1
                            with hcl.if_(dV1_dx1_L[0] > max1_deriv1[0]):
                                max1_deriv1[0] = dV1_dx1_L[0]
                            with hcl.if_(dV1_dx1_R[0] > max1_deriv1[0]):
                                max1_deriv1[0] = dV1_dx1_R[0]

                            with hcl.if_(dV1_dx2_L[0] > max1_deriv2[0]):
                                max1_deriv2[0] = dV1_dx2_L[0]
                            with hcl.if_(dV1_dx2_R[0] > max1_deriv2[0]):
                                max1_deriv2[0] = dV1_dx2_R[0]

                            with hcl.if_(dV1_dx3_L[0] > max1_deriv3[0]):
                                max1_deriv3[0] = dV1_dx3_L[0]
                            with hcl.if_(dV1_dx3_R[0] > max1_deriv3[0]):
                                max1_deriv3[0] = dV1_dx3_R[0]

                            with hcl.if_(dV1_dx4_L[0] > max1_deriv4[0]):
                                max1_deriv4[0] = dV1_dx4_L[0]
                            with hcl.if_(dV1_dx4_R[0] > max1_deriv4[0]):
                                max1_deriv4[0] = dV1_dx4_R[0]

                            # Get derivMin for player 2
                            with hcl.if_(dV2_dx1_L[0] < min2_deriv1[0]):
                                min2_deriv1[0] = dV2_dx1_L[0]
                            with hcl.if_(dV2_dx1_R[0] < min2_deriv1[0]):
                                min2_deriv1[0] = dV2_dx1_R[0]

                            with hcl.if_(dV2_dx2_L[0] < min2_deriv2[0]):
                                min2_deriv2[0] = dV2_dx2_L[0]
                            with hcl.if_(dV2_dx2_R[0] < min2_deriv2[0]):
                                min2_deriv2[0] = dV2_dx2_R[0]

                            with hcl.if_(dV2_dx3_L[0] < min2_deriv3[0]):
                                min2_deriv3[0] = dV2_dx3_L[0]
                            with hcl.if_(dV2_dx3_R[0] < min2_deriv3[0]):
                                min2_deriv3[0] = dV2_dx3_R[0]

                            with hcl.if_(dV2_dx4_L[0] < min2_deriv4[0]):
                                min2_deriv4[0] = dV2_dx4_L[0]
                            with hcl.if_(dV2_dx4_R[0] < min2_deriv4[0]):
                                min2_deriv4[0] = dV2_dx4_R[0]

                            # Get derivMax for player 2
                            with hcl.if_(dV2_dx1_L[0] > max2_deriv1[0]):
                                max2_deriv1[0] = dV2_dx1_L[0]
                            with hcl.if_(dV2_dx1_R[0] > max2_deriv1[0]):
                                max2_deriv1[0] = dV2_dx1_R[0]

                            with hcl.if_(dV2_dx2_L[0] > max2_deriv2[0]):
                                max2_deriv2[0] = dV2_dx2_L[0]
                            with hcl.if_(dV2_dx2_R[0] > max2_deriv2[0]):
                                max2_deriv2[0] = dV2_dx2_R[0]

                            with hcl.if_(dV2_dx3_L[0] > max2_deriv3[0]):
                                max2_deriv3[0] = dV2_dx3_L[0]
                            with hcl.if_(dV2_dx3_R[0] > max2_deriv3[0]):
                                max2_deriv3[0] = dV2_dx3_R[0]

                            with hcl.if_(dV2_dx4_L[0] > max2_deriv4[0]):
                                max2_deriv4[0] = dV2_dx4_L[0]
                            with hcl.if_(dV2_dx4_R[0] > max2_deriv4[0]):
                                max2_deriv4[0] = dV2_dx4_R[0]

        # Calculate dissipation amount
        with hcl.Stage("Dissipation"):
            # Storing alphas
            alpha1 = hcl.scalar(0, "alpha1")
            alpha2 = hcl.scalar(0, "alpha2")
            alpha3 = hcl.scalar(0, "alpha3")
            alpha4 = hcl.scalar(0, "alpha4")

            # Lower bound optimal control for player 1
            u1optL1 = hcl.scalar(0, "u1optL1")
            u1optL2 = hcl.scalar(0, "u1optL2")
            u1optL3 = hcl.scalar(0, "u1optL3")
            u1optL4 = hcl.scalar(0, "u1optL4")

            # Upper bound optimal control for player 1
            u1optU1 = hcl.scalar(0, "u1optU1")
            u1optU2 = hcl.scalar(0, "u1optU2")
            u1optU3 = hcl.scalar(0, "u1optU3")
            u1optU4 = hcl.scalar(0, "u1optU4")

            # Lower bound optimal control for player 2
            u2optL1 = hcl.scalar(0, "u2optL1")
            u2optL2 = hcl.scalar(0, "u2optL2")
            u2optL3 = hcl.scalar(0, "u2optL3")
            u2optL4 = hcl.scalar(0, "u2optL4")

            # Upper bound optimal control for player 2
            u2optU1 = hcl.scalar(0, "u2optU1")
            u2optU2 = hcl.scalar(0, "u2optU2")
            u2optU3 = hcl.scalar(0, "u2optU3")
            u2optU4 = hcl.scalar(0, "u2optU4")

            with hcl.for_(0, V1_init.shape[0], name="i") as i:
                with hcl.for_(0, V1_init.shape[1], name="j") as j:
                    with hcl.for_(0, V1_init.shape[2], name="k") as k:
                        with hcl.for_(0, V1_init.shape[3], name="l") as l:
                            dx_LL1 = hcl.scalar(0, "dx_LL1")
                            dx_LL2 = hcl.scalar(0, "dx_LL2")
                            dx_LL3 = hcl.scalar(0, "dx_LL3")
                            dx_LL4 = hcl.scalar(0, "dx_LL4")

                            dx_UL1 = hcl.scalar(0, "dx_UL1")
                            dx_UL2 = hcl.scalar(0, "dx_UL2")
                            dx_UL3 = hcl.scalar(0, "dx_UL3")
                            dx_UL4 = hcl.scalar(0, "dx_UL4")

                            dx_UU1 = hcl.scalar(0, "dx_UU1")
                            dx_UU2 = hcl.scalar(0, "dx_UU2")
                            dx_UU3 = hcl.scalar(0, "dx_UU3")
                            dx_UU4 = hcl.scalar(0, "dx_UU4")

                            dx_LU1 = hcl.scalar(0, "dx_LU1")
                            dx_LU2 = hcl.scalar(0, "dx_LU2")
                            dx_LU3 = hcl.scalar(0, "dx_LU3")
                            dx_LU4 = hcl.scalar(0, "dx_LU4")

                            # Find lower bound optimal control for player 1
                            u1optL1[0], u1optL2[0], u1optL3[0], u1optL4[0] = my_object.opt_u1(t, (x1[i], x2[j], x3[k], x4[l]),
                                                                             (min1_deriv1[0], min1_deriv2[0], min1_deriv3[0], min1_deriv4[0]))

                            # Find upper bound optimal control for player 1
                            u1optU1[0], u1optU2[0], u1optU3[0], u1optU4[0] = my_object.opt_u1(t, (x1[i], x2[j], x3[k], x4[l]),
                                                                             (max1_deriv1[0], max1_deriv2[0], max1_deriv3[0], max1_deriv4[0]))

                            # Find lower bound optimal control for player 2
                            u2optL1[0], u2optL2[0], u2optL3[0], u2optL4[0] = my_object.opt_u2(t, (x1[i], x2[j], x3[k], x4[l]),
                                                                             (min2_deriv1[0], min2_deriv2[0], min2_deriv3[0], min2_deriv4[0]))

                            # Find upper bound optimal control for player 2
                            u2optU1[0], u2optU2[0], u2optU3[0], u2optU4[0] = my_object.opt_u2(t, (x1[i], x2[j], x3[k], x4[l]),
                                                                             (max2_deriv1[0], max2_deriv2[0], max2_deriv3[0], max2_deriv4[0]))

                            # Find magnitude of rates of changes
                            dx_LL1[0], dx_LL2[0], dx_LL3[0], dx_LL4[0] = my_object.dynamics(t, (x1[i], x2[j], x3[k], x4[l]),
                                                                                            (u1optL1[0], u1optL2[0], u1optL3[0], u1optL4[0]),
                                                                                            (u2optL1[0], u2optL2[0], u2optL3[0], u2optL4[0]))
                            dx_LL1[0] = my_abs(dx_LL1[0])
                            dx_LL2[0] = my_abs(dx_LL2[0])
                            dx_LL3[0] = my_abs(dx_LL3[0])
                            dx_LL4[0] = my_abs(dx_LL4[0])

                            dx_LU1[0], dx_LU2[0], dx_LU3[0], dx_LU4[0] = my_object.dynamics(t, (x1[i], x2[j], x3[k], x4[l]),
                                                                                            (u1optL1[0], u1optL2[0], u1optL3[0], u1optL4[0]),
                                                                                            (u2optU1[0], u2optU2[0], u2optU3[0], u2optU4[0]))
                            dx_LU1[0] = my_abs(dx_LU1[0])
                            dx_LU2[0] = my_abs(dx_LU2[0])
                            dx_LU3[0] = my_abs(dx_LU3[0])
                            dx_LU4[0] = my_abs(dx_LU4[0])

                            # Calculate alpha
                            alpha1[0] = my_max(dx_LL1[0], dx_LU1[0])
                            alpha2[0] = my_max(dx_LL2[0], dx_LU2[0])
                            alpha3[0] = my_max(dx_LL3[0], dx_LU3[0])
                            alpha4[0] = my_max(dx_LL4[0], dx_LU4[0])

                            dx_UL1[0], dx_UL2[0], dx_UL3[0], dx_UL4[0] = my_object.dynamics(t, (x1[i], x2[j], x3[k], x4[l]),
                                                                                            (u1optU1[0], u1optU2[0], u1optU3[0], u1optU4[0]),
                                                                                            (u2optL1[0], u2optL2[0], u2optL3[0], u2optL4[0]))
                            dx_UL1[0] = my_abs(dx_UL1[0])
                            dx_UL2[0] = my_abs(dx_UL2[0])
                            dx_UL3[0] = my_abs(dx_UL3[0])
                            dx_UL4[0] = my_abs(dx_UL4[0])
                            
                            # Calculate alpha
                            alpha1[0] = my_max(alpha1[0], dx_UL1[0])
                            alpha2[0] = my_max(alpha2[0], dx_UL2[0])
                            alpha3[0] = my_max(alpha3[0], dx_UL3[0])
                            alpha4[0] = my_max(alpha4[0], dx_UL4[0])

                            dx_UU1[0], dx_UU2[0], dx_UU3[0], dx_UU4[0] = my_object.dynamics(t, (x1[i], x2[j], x3[k], x4[l]),
                                                                                            (u1optU1[0], u1optU2[0], u1optU3[0], u1optU4[0]),
                                                                                            (u2optU1[0], u2optU2[0], u2optU3[0], u2optU4[0]))
                            dx_UU1[0] = my_abs(dx_UU1[0])
                            dx_UU2[0] = my_abs(dx_UU2[0])
                            dx_UU3[0] = my_abs(dx_UU3[0])
                            dx_UU4[0] = my_abs(dx_UU4[0])

                            # Calculate alphas
                            alpha1[0] = my_max(alpha1[0], dx_UU1[0])
                            alpha2[0] = my_max(alpha2[0], dx_UU2[0])
                            alpha3[0] = my_max(alpha3[0], dx_UU3[0])
                            alpha4[0] = my_max(alpha4[0], dx_UU4[0])

                            diss1 = hcl.scalar(0, "diss1")
                            diss1[0] = 0.5 * (deriv1_diff1[i, j, k, l] * alpha1[0] + deriv1_diff2[i, j, k, l] * alpha2[0] +
                                              deriv1_diff3[i, j, k, l] * alpha3[0] + deriv1_diff4[i, j, k, l] * alpha4[0])
                            diss2 = hcl.scalar(0, "diss2")
                            diss2[0] = 0.5 * (deriv2_diff1[i, j, k, l] * alpha1[0] + deriv2_diff2[i, j, k, l] * alpha2[0] +
                                              deriv2_diff3[i, j, k, l] * alpha3[0] + deriv2_diff4[i, j, k, l] * alpha4[0])
                            #probe[i, j, k, l] = alpha1[0]

                            # Finally
                            V1_new[i, j, k, l] = -(V1_new[i, j, k, l] - diss1[0])
                            V2_new[i, j, k, l] = -(V2_new[i, j, k, l] - diss2[0])
                            
                            # Get maximum alphas in each dimension
                            with hcl.if_(alpha1[0] > max_alpha1[0]):
                                max_alpha1[0] = alpha1[0]
                            with hcl.if_(alpha2[0] > max_alpha2[0]):
                                max_alpha2[0] = alpha2[0]
                            with hcl.if_(alpha3[0] > max_alpha3[0]):
                                max_alpha3[0] = alpha3[0]
                            with hcl.if_(alpha4[0] > max_alpha4[0]):
                                max_alpha4[0] = alpha4[0]

        # Determine time step
        delta_t = hcl.compute((1,), lambda x: step_bound(), name="delta_t")
        # Integrate
        hcl.update(V1_new, lambda i, j, k, l: V1_init[i, j, k, l] + V1_new[i, j, k, l] * delta_t[0])
        hcl.update(V2_new, lambda i, j, k, l: V2_init[i, j, k, l] + V2_new[i, j, k, l] * delta_t[0])

        # Copy V_new to V_init
        hcl.update(V1_init, lambda i, j, k, l: V1_new[i, j, k, l])
        hcl.update(V2_init, lambda i, j, k, l: V2_new[i, j, k, l])

    def returnDerivative(V1_array, Deriv1_array, V2_array, Deriv2_array):
        with hcl.Stage("ComputeDeriv"):
            with hcl.for_(0, V1_array.shape[0], name="i") as i:
                with hcl.for_(0, V1_array.shape[1], name="j") as j:
                    with hcl.for_(0, V1_array.shape[2], name="k") as k:
                        with hcl.for_(0, V1_array.shape[3], name="l") as l:
                            dV1_dx_L = hcl.scalar(0, "dV1_dx_L")
                            dV1_dx_R = hcl.scalar(0, "dV1_dx_R")
                            dV2_dx_L = hcl.scalar(0, "dV2_dx_L")
                            dV2_dx_R = hcl.scalar(0, "dV2_dx_R")
                            if accuracy == "low":
                                if deriv_dim == 1:
                                    dV1_dx_L[0], dV1_dx_R[0] = spa_derivX1_4d(i, j, k, l, V1_array, g)
                                    dV2_dx_L[0], dV2_dx_R[0] = spa_derivX1_4d(i, j, k, l, V2_array, g)
                                if deriv_dim == 2:
                                    dV1_dx_L[0], dV1_dx_R[0] = spa_derivX2_4d(i, j, k, l, V1_array, g)
                                    dV2_dx_L[0], dV2_dx_R[0] = spa_derivX2_4d(i, j, k, l, V2_array, g)
                                if deriv_dim == 3:
                                    dV1_dx_L[0], dV1_dx_R[0] = spa_derivX3_4d(i, j, k, l, V1_array, g)
                                    dV2_dx_L[0], dV2_dx_R[0] = spa_derivX3_4d(i, j, k, l, V2_array, g)
                                if deriv_dim == 4:
                                    dV1_dx_L[0], dV1_dx_R[0] = spa_derivX4_4d(i, j, k, l, V1_array, g)
                                    dV2_dx_L[0], dV2_dx_R[0] = spa_derivX4_4d(i, j, k, l, V2_array, g)
                            if accuracy == "medium":
                                if deriv_dim == 1:
                                    dV1_dx_L[0], dV1_dx_R[0] = secondOrderX1_4d(i, j, k, l, V1_array, g)
                                    dV2_dx_L[0], dV2_dx_R[0] = secondOrderX1_4d(i, j, k, l, V2_array, g)
                                if deriv_dim == 2:
                                    dV1_dx_L[0], dV1_dx_R[0] = secondOrderX2_4d(i, j, k, l, V1_array, g)
                                    dV2_dx_L[0], dV2_dx_R[0] = secondOrderX2_4d(i, j, k, l, V2_array, g)
                                if deriv_dim == 3:
                                    dV1_dx_L[0], dV1_dx_R[0] = secondOrderX3_4d(i, j, k, l, V1_array, g)
                                    dV2_dx_L[0], dV2_dx_R[0] = secondOrderX3_4d(i, j, k, l, V2_array, g)
                                if deriv_dim == 4:
                                    dV1_dx_L[0], dV1_dx_R[0] = secondOrderX4_4d(i, j, k, l, V1_array, g)
                                    dV2_dx_L[0], dV2_dx_R[0] = secondOrderX4_4d(i, j, k, l, V2_array, g)

                            Deriv1_array[i, j, k, l] = (dV1_dx_L[0] + dV1_dx_R[0]) / 2
                            Deriv2_array[i, j, k, l] = (dV2_dx_L[0] + dV2_dx_R[0]) / 2

    if generate_SpatDeriv == False:
        s = hcl.create_schedule([V1_f, V1_init, V2_f, V2_init, x1, x2, x3, x4, t, probe], graph_create)

        ##################### CODE OPTIMIZATION HERE ###########################
        print("Optimizing\n")

        # Accessing the hamiltonian and dissipation stage
        s_H = graph_create.Hamiltonian
        s_D = graph_create.Dissipation

        # Thread parallelize hamiltonian and dissipation
        s[s_H].parallel(s_H.i)
        s[s_D].parallel(s_D.i)

        # Inspect IR
        # if args.llvm:
        #    print(hcl.lower(s))
    else:
        s = hcl.create_schedule([V1_init, V1_f, V2_init, V2_f], returnDerivative)

    # Return executable
    return (hcl.build(s))