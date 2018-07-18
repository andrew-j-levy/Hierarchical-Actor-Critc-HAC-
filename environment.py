from tkinter import *
from tkinter import ttk
import time
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer

class Environment():

    def __init__(self, file_name, subgoal_bounds, projection_fn, initial_state_space, goal_space_train, goal_space_test, goal_thresholds, max_actions, show):

        self.name = file_name

        # Create Mujoco Simulation
        self.model = load_model_from_path("./mujoco_files/" + file_name)
        self.sim = MjSim(self.model)

        # Set dimensions and ranges of states, actions, and goals in order to configure actor/critic networks
        self.state_dim = 2*len(self.sim.data.qpos) + len(self.sim.data.qvel) # State will include (i) sin() and cos() of all joint angles and (ii) joint velocities 
        self.action_dim = len(self.sim.model.actuator_ctrlrange)
        self.action_bounds = self.sim.model.actuator_ctrlrange[:,1] 
        self.action_offset = np.zeros((len(self.action_bounds))) # Assumes symmetric low-level action ranges
        self.goal_dim = len(goal_space_test)
        self.subgoal_bounds = subgoal_bounds
        self.projection_fn = projection_fn

        # Convert subgoal bounds to symmetric bounds and offset.  Need these to properly configure subgoal actor networks
        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))

        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0])/2
            self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]


        self.goal_thresholds = goal_thresholds

        # Set inital state and goal state spaces
        self.initial_state_space = initial_state_space
        self.goal_space_train = goal_space_train
        self.goal_space_test = goal_space_test
        self.subgoal_colors = ["Magenta","Green","Red","Blue","Cyan","Orange","Maroon","Gray","White","Black"] 

        self.max_actions = max_actions

        # Implement visualization if necessary
        self.visualize = show  # Visualization boolean
        if self.visualize:
            self.viewer = MjViewer(self.sim)
        self.num_frames_skip = 10
        

    # Get state, which concatenates joint positions and velocities
    def get_state(self):

        return np.concatenate([np.cos(self.sim.data.qpos),np.sin(self.sim.data.qpos),
                               self.sim.data.qvel])

    # Reset simulation to state within initial state specified by user
    def reset_sim(self):

        # Reset joint positions and velocities
        for i in range(len(self.sim.data.qpos)):
            self.sim.data.qpos[i] = np.random.uniform(self.initial_state_space[i][0],self.initial_state_space[i][1])      

        for i in range(len(self.sim.data.qvel)):
            self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],self.initial_state_space[len(self.sim.data.qpos) + i][1])       
        
        self.sim.step()

        # Return state
        return self.get_state()

    # Execute low-level action for number of frames specified by num_frames_skip
    def execute_action(self, action):

        self.sim.data.ctrl[:] = action
        for _ in range(self.num_frames_skip):
            self.sim.step()
            if self.visualize:
                self.viewer.render()

        return self.get_state()


    # Function returns an end goal
    def get_next_goal(self,test):
    
        end_goal = np.zeros((len(self.goal_space_test)))

        # Difficult to create goal ranges for pendulum task so hard coded function below 
        if self.name == "pendulum.xml" and not test:
            angle_targ = np.random.uniform(-np.pi/8,np.pi/8)
            x_coord = 0.5*np.sin(angle_targ)
            z_coord = 0.5*np.cos(angle_targ) + 0.6
            velo_target = np.random.uniform(-0.2,0.2)
            end_goal = np.array([x_coord,z_coord,velo_target])  

        elif not test and goal_space_train is not None:
            for i in range(len(self.goal_space_train)):
                end_goal[i] = np.random.uniform(self.goal_space_train[i][0],self.goal_space_train[i][1])
        else:
            assert self.goal_space_test is not None, "Need goal space for testing. Set goal_space_test variable in \"design_env.py\" file"
            
            for i in range(len(self.goal_space_test)):
                end_goal[i] = np.random.uniform(self.goal_space_test[i][0],self.goal_space_test[i][1])
                

        # Visualize End Goal
        self.sim.data.mocap_pos[0] = np.array([end_goal[0],0,end_goal[1]])   

        return end_goal
     

    # Visualize all subgoals
    def display_subgoals(self,subgoals):

        # Display up to 10 subgoals and end goal
        if len(subgoals) <= 11:
            subgoal_ind = 0
        else:
            subgoal_ind = len(subgoals) - 11

        for i in range(1,min(len(subgoals),11)):
            self.sim.data.mocap_pos[i] = np.array([subgoals[subgoal_ind][0],0,subgoals[subgoal_ind][1]])
            # Visualize subgoal
            self.sim.model.site_rgba[i][3] = 1
            subgoal_ind += 1

    
    # Convert state to subgoal space
    def convert_state_2_subgoal(self,state):
        return self.projection_fn(state)
        
    
    


    




