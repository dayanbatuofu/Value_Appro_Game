##### Change this line for different a different problem. #####

system = 'vehicle'

time_dependent = True  # True

if system == 'vehicle':
    from examples.vehicle.problem_def_intersection import setup_problem, config_NN

problem = setup_problem()
config = config_NN(problem.N_states, time_dependent)
