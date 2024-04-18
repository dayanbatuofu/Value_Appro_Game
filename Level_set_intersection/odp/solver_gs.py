import heterocl as hcl
import numpy as np
import time

# Backward reachable set computation library
from odp.computeGraphs import graph_4D_gs

def HJSolver(dynamics_obj, grid, multiple_value, tau, saveAllTimeSteps=False,
             accuracy="lower", untilConvergent=False, epsilon=2e-3):

    print("Welcome to optimized_dp \n")

    init_value1 = multiple_value[0]  # Target value for player 1
    init_value2 = multiple_value[1]  # Target value for player 2
    
    hcl.init()
    hcl.config.init_dtype = hcl.Float(32)

    ################# INITIALIZE DATA TO BE INPUT INTO EXECUTABLE ##########################

    print("Initializing\n")

    # Tensors input to our computation graph
    # V1_0 = hcl.asarray(init_value1)
    V1_0 = hcl.asarray(np.zeros(tuple(grid.pts_each_dim)))
    V1_1 = hcl.asarray(np.zeros(tuple(grid.pts_each_dim)))

    # V2_0 = hcl.asarray(init_value2)
    V2_0 = hcl.asarray(np.zeros(tuple(grid.pts_each_dim)))
    V2_1 = hcl.asarray(np.zeros(tuple(grid.pts_each_dim)))

    # For debugging purposes
    probe = hcl.asarray(np.zeros(tuple(grid.pts_each_dim)))

    # Array for each state values
    list_x1 = np.reshape(grid.vs[0], grid.pts_each_dim[0])
    list_x2 = np.reshape(grid.vs[1], grid.pts_each_dim[1])
    list_x3 = np.reshape(grid.vs[2], grid.pts_each_dim[2])
    list_x4 = np.reshape(grid.vs[3], grid.pts_each_dim[3])

    # Convert state arrays to hcl array type
    list_x1 = hcl.asarray(list_x1)
    list_x2 = hcl.asarray(list_x2)
    list_x3 = hcl.asarray(list_x3)
    list_x4 = hcl.asarray(list_x4)

    # Get executable, obstacle check intial value function
    solve_pde = graph_4D_gs(dynamics_obj, grid, accuracy)

    """ Be careful, for high-dimensional array (5D or higher), saving value arrays at all the time steps may 
    cause your computer to run out of memory """
    if saveAllTimeSteps is True:
        valfuncs1 = np.zeros(np.insert(tuple(grid.pts_each_dim), grid.dims, len(tau)))
        valfuncs1[..., -1] = V1_0.asnumpy()
        valfuncs2 = np.zeros(np.insert(tuple(grid.pts_each_dim), grid.dims, len(tau)))
        valfuncs2[..., -1] = V2_0.asnumpy()
        print(valfuncs2.shape)

    ################ USE THE EXECUTABLE ############
    # Variables used for timing
    execution_time = 0
    iter = 0
    tNow = tau[0]
    print("Started running\n")

    # Backward reachable set/tube will be computed over the specified time horizon
    # Or until convergent ( which ever happens first )
    for i in range(1, len(tau)):
        t_minh = hcl.asarray(np.array((tNow, tau[i])))

        while tNow <= tau[i] - 1e-4:
            prev1_arr = V1_0.asnumpy()
            prev2_arr = V2_0.asnumpy()
            # Start timing
            iter += 1
            start = time.time()

            # Run the execution and pass input into graph
            solve_pde(V1_1, V1_0, V2_1, V2_0, list_x1, list_x2, list_x3, list_x4, t_minh, probe)

            tNow = t_minh.asnumpy()[0]

            # Calculate computation time
            execution_time += time.time() - start

            # Some information printin
            print(t_minh)
            print("Computational time to integrate (s): {:.5f}".format(time.time() - start))

            if untilConvergent is True:
                # Compare difference between V_{t-1} and V_{t} and choose the max changes
                diff1 = np.amax(np.abs(V1_1.asnumpy() - prev1_arr))
                diff2 = np.amax(np.abs(V2_1.asnumpy() - prev2_arr))
                print("Max difference between V_old and V_new for player 1: {:.5f}".format(diff1))
                print("Max difference between V_old and V_new for player 2: {:.5f}".format(diff2))
                if diff1 < epsilon and diff2 < epsilon:
                    print("Result converged ! Exiting the compute loop. Have a good day.")
                    break
        else:  # if it didn't break because of convergent condition
            if saveAllTimeSteps is True:
                valfuncs1[..., -1-i] = V1_1.asnumpy()
                valfuncs2[..., -1-i] = V2_1.asnumpy()
                # pass
            continue
        break  # only if convergent condition is achieved

    # Time info printing
    print("Total kernel time (s): {:.5f}".format(execution_time))
    print("Finished solving\n")

    if saveAllTimeSteps is True:
        valfuncs1[..., 0] = V1_1.asnumpy()
        valfuncs2[..., 0] = V2_1.asnumpy()
        return valfuncs1, valfuncs2

    return V1_1.asnumpy(), V2_1.asnumpy()

def computeSpatDerivArray(grid, V1, V2, deriv_dim, accuracy="low"):
    # Return a tensor same size as V that contains spatial derivatives at every state in V
    hcl.init()
    hcl.config.init_dtype = hcl.Float(32)

    # Need to make sure that value array has the same size as grid
    assert list(V1.shape) == list(grid.pts_each_dim)

    V1_0 = hcl.asarray(V1)
    V2_0 = hcl.asarray(V2)
    spatial_deriv1 = hcl.asarray(np.zeros(tuple(grid.pts_each_dim)))
    spatial_deriv2 = hcl.asarray(np.zeros(tuple(grid.pts_each_dim)))

    # Get executable, obstacle check intial value function
    compute_SpatDeriv = graph_4D_gs(None, grid, accuracy, generate_SpatDeriv=True, deriv_dim=deriv_dim)

    compute_SpatDeriv(V1_0, spatial_deriv1, V2_0, spatial_deriv2)
    return spatial_deriv1.asnumpy(), spatial_deriv2.asnumpy()
