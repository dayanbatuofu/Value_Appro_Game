# Value Approximation for Two-Player General-Sum Differential Games with State Constraints
<br>
Lei Zhang,
Mukesh Ghimire, 
Wenlong Zhang, 
Zhe Xu, 
Yi Ren<br>
Arizona State University

This is the accepted T-RO paper <a href="https://arxiv.org/pdf/2311.16520.pdf"> "Value Approximation for Two-Player General-Sum Differential Games with State Constraints"</a>

## Get started
There exists three different environment, you can set up a conda environment with all dependencies like so:

For Toy_case, Uncontrolled_intersection_complete_information_game, Uncontrolled_intersection_incomplete_information_game,  Narrow_road_collision_avoidance, Double_lane_change, Two_drone_collision_avoidance
```
conda env create -f environment.yml
conda activate siren
```
For BVP_generation
```
conda env create -f environment.yml
conda activate hji
```
For level_set_intersection
```
conda env create -f environment.yml
conda activate OptimizedDP
```

## Code structure
There are eight folders with different functions
### BVP_generation: use standard BVP solver to collect the Nash equilibrial (NE) values for uncontrolled intersection (case 1), narrow arod collision avoidance (case 2), double lane change (case 3) and two-drone collision avoidance (case 4)
The code is organized as follows:
* generate_intersection.py: generate 5D NE values functions under four player type configurations (a, a), (na, a), (a, na), and (na, na) for case 1.
* generate_narrow_road.py: generate 9D NE values functions for case 2.
* generate_lane_change.py: generate 9D NE values functions for case 3.
* generate_drone_avoidance.py: generate 13D NE values functions for case 4.
* ./utilities/BVP_solver.py: BVP solver.
* ./example/vehicle/problem_def_intersection.py: dynamic, PMP equation setting for case 1.
* ./example/vehicle/problem_def_narrow_road.py: dynamic, PMP equation setting for case 2.
* ./example/vehicle/problem_def_lane_change.py: dynamic, PMP equation setting for case 3.
* ./example/vehicle/problem_def_drone_avoidance.py: dynamic, PMP equation setting for case 4.

run `generate_intersection.py`, `generate_narrow_road.py`, `generate_lane_change.py`, `generate_drone_avoidance.py` to collect NE values. Please notice there is four player types in case 1. You should give setting in `generate_intersection.py`. Data size can be set in `./example/vehicle/problem_def_intersection.py`, `./example/vehicle/problem_def_narrow_road.py`, `./example/vehicle/problem_def_lane_change.py`, `./example/vehicle/problem_def_drone_avoidance.py`

### Toy Case: train supervised(SL), physics-informed neural network(PINN), hybrid(HL) and value hardening(VH) model to visualize the toy case shown in the paper
The code is organized as follows:
* dataio.py: load training data for SL, PINN, HL and VH.
* training_supervised.py: contains SL training routine.
* training_selfsupervised.py: contains PINN training routine.
* training_hybrid.py: contains HL training routine.
* training_supervised.py: contains SL training routine.
* training_valuehardening.py: contains VH training routine.
* loss_functions.py: contains loss functions for SL, PINN, HL and VH.
* modules.py: contains layers and full neural network modules.
* utils.py: contains utility functions.
* diff_operators.py: contains implementations of differential operators.
* ./experiment_scripts/toy_case.py: contains scripts to train the model, which can reproduce experiments in the paper.
* ./validation_scripts/toy_case_value_plot.py: use this script to plot Fig 1(a) in the paper. 
* ./validation_scripts/toy_case_value_hardening_plots.py: use this script to plot Fig 1(b) in the paper. 

### Uncontrolled_intersection_complete_information_game/Narrow_road_collision_avoidance/Double_lane_change/Two_drone_collision_avoidance: train supervised(SL), physics-informed neural network(PINN), hybrid(HL), epigraphical(EL) and value hardening(VH) model to complete generalization and saftety performance test for case 1/2/3/4 with complete information
The code is organized as follows:
* dataio.py: load training data for SL, PINN, HL, EL and VH.
* training_supervised.py: contains SL training routine.
* training_pinn.py: contains PINN training routine.
* training_hybrid.py: contains HL training routine.
* training_supervised.py: contains SL training routine.
* training_valuehardening.py: contains VH training routine.
* training_epigraphical.py: contains EL training routine.
* loss_functions.py: contains loss functions for SL, PINN, HL, EL and VH (case 4 only includes SL, PINN and HL).
* modules.py: contains layers and full neural network modules.
* modules_adaptive.py: contains layers and full neural network modules with adaptive activation function for EL training.
* utils.py: contains utility functions.
* diff_operators.py: contains implementations of differential operators.
* sim_draw_complete.py: animation of one case for SL, PINN, HL, EL and VH for case 1, reader can dirctly run and watch.
* sim_draw_transparent_complete.py: visualization of one case for SL, PINN, HL, EL and VH for case 1, reader can dirctly run and watch.
* sim_draw_incomplete.py: animation of one case for SL, PINN, HL, EL and VH for case 1, reader can dirctly run and watch.
* sim_draw_transparent_incomplete.py: visualization of one case for SL, PINN, HL, EL and VH for case 1, reader can dirctly run and watch.
* sim_draw_narrowroad.py: animation of one case for SL, PINN, HL, EL and VH for case 2, reader can dirctly run and watch.
* sim_draw_transparent_narrowroad.py: visualization of one case for SL, PINN, HL, EL and VH for case 2, reader can dirctly run and watch.
* sim_draw_lanechange.py: animation of one case for SL, PINN, HL, EL and VH for case 3, reader can dirctly run and watch.
* sim_draw_transparent_lanechange.py: visualization of one case for SL, PINN, HL, EL and VH for case 3, reader can dirctly run and watch.
* ./experiment_scripts/train_intersection_HJI.py: contains scripts to train the model, which can reproduce experiments in the paper.
* ./validation_scripts/closedloop_traj_generation_tanh.py: use value network (tanh as activation function) as closed-loop controllers to generate data including generalization and saftety performance.
* ./validation_scripts/closedloop_traj_generation_relu.py: use value network (relu as activation function) as closed-loop controllers to generate data including generalization and saftety performance.
* ./validation_scripts/closedloop_traj_generation_sine.py: use value network (sine as activation function) as closed-loop controllers to generate data including generalization and saftety performance.
* ./validation_scripts/closedloop_traj_generation_gelu.py: use value network (gelu as activation function) as closed-loop controllers to generate data including generalization and saftety performance (only for case 1).
* ./validation_scripts/closedloop_traj_epigraphical_tanh.py: use EL value network (tanh as activation function) as closed-loop controllers to generate data including generalization and saftety performance.
* ./validation_scripts/closedloop_traj_epigraphical_relu.py: use EL value network (relu as activation function) as closed-loop controllers to generate data including generalization and saftety performance.
* ./validation_scripts/closedloop_traj_epigraphical_sine.py: use EL value network (sine as activation function) as closed-loop controllers to generate data including generalization and saftety performance.
* ./validation_scripts/trajectory_with_value_tanh.py: visualize generalization and saftety performance for value network (tanh as activation function).
* ./validation_scripts/trajectory_with_value_relu.py: visualize generalization and saftety performance for value network (relu as activation function).
* ./validation_scripts/trajectory_with_value_sine.py: visualize generalization and saftety performance for value network (sine as activation function).
* ./validation_scripts/trajectory_with_value_gelu.py: visualize generalization and saftety performance for value network (gelu as activation function, only for case 1).
* ./validation_scripts/value_generation_tanh.py: measure the MAEs of value and control input predictions across the test trajectories (only for case 1).
* ./validation_scripts/action_compute_tanh_initial state.py: present measure the MAEs of control input prediction for initial state space (only for case 1).
* ./validation_scripts/action_compute_tanh_expanded state.py: present measure the MAEs of control input prediction for expanded state space (only for case 1).
* ./validation_scripts/value_compute_tanh_initial state.py: present measure the MAEs of value prediction for initial state space (only for case 1).
* ./validation_scripts/value_compute_tanh_expanded state.py: present measure the MAEs of value prediction for expanded state space (only for case 1).
* ./validation_scripts/valuecontour_generate_epigraphical.py: generate value contour for EL method.
* ./validation_scripts/valuecontour_generate_hybrid.py: generate value contour for HL method.
* ./validation_scripts/valuecontour_gt_plot.py: plot value contour in Fig 4. Download the ground truth data before plotting: <a href="https://drive.google.com/file/d/1KGnf94P3VrXH3JRKBiz5ME1Y2WYy4Mzi/view?usp=sharing"> link. 
* ./validation_scripts/frequency_generation.py: implement FFT for value approximation in Case 1,2,3.
* ./validation_scripts/frequency_plot.py: plot FFT of approximated value in d1-d2 domina, shown in Fig 13.
* ./validation_scripts/model: experimental model in the paper.
* ./validation_scripts/train_data: training data in the paper.
* ./validation_scripts/test_data: testing data in the paper.
* ./validation_scripts/closed_loop: store data by using value network as closed-loop controllers.
* ./validation_scripts/value: store data to measure MAE of value and control input predictions.

### Uncontrolled_intersection_complete_information_game: use supervised(SL) and hybrid(HL) model to complete incomplete information games for case 1
The code is organized as follows:
* main.py: run the simulatio with initial setting including agent's belief, empathetic or non-empathetic.
* enviroment.py: simulation environment is generated here, using the parameters from main.py.
* savi_simulation.py: initial conditions are processed for the simulation here, such as agent parameters (beta) and action set. The initialization belief table is also done here through the function get_initial_belief().
* inference_model.py: inference is done after observing the state. There are several models implemented here: bvp, baseline, etc. The inference algorithm updates the belief table at each time step using the selected model defined in main.py.
* decision.py: decision model returns an action for each agent, depending on the type of agent defined in main.py. Models include bvp_empathetic, bvp_non_empathetic, baseline, etc.
* plot_loss_traj.py: convert .cvs file into .mat file to generate two-player trajectories projected into d1-d2 frame. 
* modules.py: contains layers and full neural network modules.
* utils.py: contains utility functions.
* diff_operators.py: contains implementations of differential operators.
* ./experiment/store the data of simulation.
* ./validation_scripts/model: experimental model in the paper.
* ./validation_scripts/Hamilton_generation.py: use value network to predict the state in the simulation.
* ./validation_scripts/trajectory_policy_consistent.py: plot two-player trajectories projected into d1-d2 frame when players' initla belief is consistent with their true parameter. 
* ./validation_scripts/trajectory_policy_consistent.py: plot two-player trajectories projected into d1-d2 frame when players' initla belief is not consistent with their true parameter. 

### level_set_intersection: extend existing level set methods from zero-sum games to general-sum games to compute HJI values
The code is organized as follows, exsting solver: <a href="https://github.com/SFU-MARS/optimized_dp"> link
* ./examples/examples_5D.py: uncontrolled intersection problem setting.
* ./op/computeGraphs/graph_4D_gs.py: mesh grid computation
* ./op/dynamics/Intersection.py: dynamics setting
* ./op/Grid/GridProcessing.py: how to set up mesh grid.
* ./op/Shape/ShapesFunctions.py: shape function setting for uncontrolled intersection.
* ./op/spatialDerivatives/first_orderENO4D.py: first order derivative computation.
* ./op/spatialDerivatives/second_orderENO4D.py: second order derivative computation.

## Contact
If you have any questions, please feel free to email the authors.
