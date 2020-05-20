import numpy as np
from layer import Layer
from environment import Environment
import pickle as cpickle
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import pickle as cpickle

# Below class instantiates an agent
class Agent():
    def __init__(self,FLAGS, env, agent_params):

        self.FLAGS = FLAGS
        self.sess = tf.Session()

        # Set subgoal testing ratio each layer will use
        self.subgoal_test_perc = agent_params["subgoal_test_perc"]

        # Create agent with number of levels specified by user
        self.layers = [Layer(i,FLAGS,env,self.sess,agent_params) for i in range(FLAGS.layers)]

        # Below attributes will be used help save network parameters
        self.saver = None
        self.model_dir = None
        self.model_loc = None

        # Initialize actor/critic networks.  Load saved parameters if not retraining
        self.initialize_networks()

        # goal_array will store goal for each layer of agent.
        self.goal_array = [None for i in range(FLAGS.layers)]

        self.current_state = None

        # Track number of low-level actions executed
        self.steps_taken = 0

        # Below hyperparameter specifies number of Q-value updates made after each episode
        self.num_updates = 40

        # Below parameters will be used to store performance results
        self.performance_log = []

        self.other_params = agent_params


    # Determine whether or not each layer's goal was achieved.  Also, if applicable, return the highest level whose goal was achieved.
    def check_goals(self,env):

        # goal_status is vector showing status of whether a layer's goal has been achieved
        goal_status = [False for i in range(self.FLAGS.layers)]

        max_lay_achieved = None

        # Project current state onto the subgoal and end goal spaces
        proj_subgoal = env.project_state_to_subgoal(env.sim, self.current_state)
        proj_end_goal = env.project_state_to_end_goal(env.sim, self.current_state)

        for i in range(self.FLAGS.layers):

            goal_achieved = True

            # If at highest layer, compare to end goal thresholds
            if i == self.FLAGS.layers - 1:

                # Check dimensions are appropriate
                assert len(proj_end_goal) == len(self.goal_array[i]) == len(env.end_goal_thresholds), "Projected end goal, actual end goal, and end goal thresholds should have same dimensions"

                # Check whether layer i's goal was achieved by checking whether projected state is within the goal achievement threshold
                for j in range(len(proj_end_goal)):
                    if np.absolute(self.goal_array[i][j] - proj_end_goal[j]) > env.end_goal_thresholds[j]:
                        goal_achieved = False
                        break

            # If not highest layer, compare to subgoal thresholds
            else:

                # Check that dimensions are appropriate
                assert len(proj_subgoal) == len(self.goal_array[i]) == len(env.subgoal_thresholds), "Projected subgoal, actual subgoal, and subgoal thresholds should have same dimensions"

                # Check whether layer i's goal was achieved by checking whether projected state is within the goal achievement threshold
                for j in range(len(proj_subgoal)):
                    if np.absolute(self.goal_array[i][j] - proj_subgoal[j]) > env.subgoal_thresholds[j]:
                        goal_achieved = False
                        break

            # If projected state within threshold of goal, mark as achieved
            if goal_achieved:
                goal_status[i] = True
                max_lay_achieved = i
            else:
                goal_status[i] = False


        return goal_status, max_lay_achieved


    def initialize_networks(self):

        model_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(model_vars)

        # Set up directory for saving models
        self.model_dir = os.getcwd() + '/models'
        self.model_loc = self.model_dir + '/HAC.ckpt'

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

         # Initialize actor/critic networks
        self.sess.run(tf.global_variables_initializer())

        # If not retraining, restore weights
        # if we are not retraining from scratch, just restore weights
        if self.FLAGS.retrain == False:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))


    # Save neural network parameters
    def save_model(self, episode):
        self.saver.save(self.sess, self.model_loc, global_step=episode)


    # Update actor and critic networks for each layer
    def learn(self):

        for i in range(len(self.layers)):
            self.layers[i].learn(self.num_updates)


    # Train agent for an episode
    def train(self,env, episode_num,total_episodes):

        # Select final goal from final goal space, defined in "design_agent_and_env.py"
        self.goal_array[self.FLAGS.layers - 1] = env.get_next_goal(self.FLAGS.test)
        print("Next End Goal: ", self.goal_array[self.FLAGS.layers - 1])

        # Select initial state from in initial state space, defined in environment.py
        self.current_state = env.reset_sim(self.goal_array[self.FLAGS.layers - 1])
        if env.name == "ant_reacher.xml":
            print("Initial Ant Position: ", self.current_state[:3])
        # print("Initial State: ", self.current_state)

        # Reset step counter
        self.steps_taken = 0

        # Train for an episode
        goal_status, max_lay_achieved = self.layers[self.FLAGS.layers-1].train(self,env, episode_num = episode_num)

        # Update actor/critic networks if not testing
        if not self.FLAGS.test and total_episodes > self.other_params["num_pre_training_episodes"]:
            self.learn()

        # Return whether end goal was achieved
        return goal_status[self.FLAGS.layers-1]


    # Save performance evaluations
    def log_performance(self, success_rate):

        # Add latest success_rate to list
        self.performance_log.append(success_rate)

        # Save log
        cpickle.dump(self.performance_log,open("performance_log.p","wb"))
