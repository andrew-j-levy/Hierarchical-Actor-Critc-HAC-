"""
Below function designs the training environment.  The designer must provide (i) a Mujoco model (i.e., .xml file), (ii) subgoal space, (iii) state to subgoal projection function (if subgoal is different than state), (iv) initial state space, (v) final goal state space, (vi) goal achievement thresholds, and (vii) maximum episode length.
"""

import numpy as np
from environment import Environment

def design_env(show):

    # Provide file name of Mujoco model(i.e., "pendulum.xml").  Make sure file is stored in "mujoco_files" folder
    model_name = "pendulum.xml"
    
    # Provide range for each dimension of subgoal space in order to configure subgoal actor networks.  Subgoal space can be the same as the state space or some other projection out of the state space.  For instance, in our implementation of pendulum, the subgoal space is [pendulum x_pos, pendulum y_pos, angular velocity], which is slightly different than the state space [cos(pendulum angle), sin(pendulum angle), angular velocity].
    subgoal_bounds = np.array([[-0.5,0.5],[0.1,1.1],[-15,15]]) 

    # Provide state to subgoal projection function if subgoal space is different than state space.  This will be important for hindsight learning as agent will need to know what subgoal it has achieved when it has failed to achieve its original subgoal.  I
    projection_fn = lambda state: np.array([0.5*state[1],0.5*state[0] + 0.6, 15 if state[2] > 15 else -15 if state[2] < -15 else state[2]])   

    # Range for each dimension of initial state and end goal state spaces.  In pendulum, below variable simply holding the initial joint angle and velocity of the pendulum, which is slightly different than the actual state space  
    initial_state_space = np.array([[np.pi/4, 7*np.pi/4],[-0.05,0.05]])

    # Supports two types of end goal spaces if user would like to train on a larger end goal space.  For pendulum, it is difficult to specify position ranges as the pendulum always lies on a circle.  The training goal space is thus hard-coded in the get_next_goal method in "environment.py".
    goal_space_train = None
    goal_space_test = [[0,0],[1.1,1.1],[0,0]]
        

    # Set goal achievement thresholds.  If the agent is within the threshold for each dimension, the subgoal or end goal has been achieved.
    goal_thresholds = [0.075, 0.075, 0.75]  

    # Provide maximum episode length in terms of the maximum number of low-level actions allowed in an episode
    max_actions = 1200


    # Ensure upper bounds of range is >= lower bound of range
    for i in range(len(subgoal_bounds)):
        assert subgoal_bounds[i][1] >= subgoal_bounds[i][0], "In subgoal space, upper bound must be >= lower bound"

    for i in range(len(initial_state_space)):
        assert initial_state_space[i][1] >= initial_state_space[i][0], "In initial state space, upper bound must be >= lower bound"

    if goal_space_train is not None:
        for i in range(len(goal_space_train)):
            assert goal_space_train[i][1] >= goal_space_train[i][0], "In the training goal space, upper bound must be >= lower bound"

    if goal_space_test is not None:
        for i in range(len(goal_space_test)):
            assert goal_space_test[i][1] >= goal_space_test[i][0], "In the training goal space, upper bound must be >= lower bound"

    # Make sure two goal spaces have same dimensions
    if goal_space_train is not None and goal_space_test is not None:
        assert len(goal_space_train) == len(goal_space_test), "Traing and testing goal spaces must have same dimensions"

    
    return Environment(model_name, subgoal_bounds, projection_fn, initial_state_space, goal_space_train, goal_space_test, goal_thresholds, max_actions, show)
    
