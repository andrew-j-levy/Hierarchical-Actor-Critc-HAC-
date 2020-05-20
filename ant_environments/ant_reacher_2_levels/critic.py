import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from utils import layer

class Critic():

    def __init__(self, sess, env, layer_number, FLAGS, learning_rate=0.001, gamma=0.98, tau=0.05):
        self.sess = sess
        self.critic_name = 'critic_' + str(layer_number)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
       
        self.q_limit = -FLAGS.time_scale

        # Dimensions of goal placeholder will differ depending on layer level
        if layer_number == FLAGS.layers - 1:
            self.goal_dim = env.end_goal_dim
        else:
            self.goal_dim = env.subgoal_dim

        self.loss_val = 0
        self.state_dim = env.state_dim
        self.state_ph = tf.placeholder(tf.float32, shape=(None, env.state_dim), name='state_ph')
        self.goal_ph = tf.placeholder(tf.float32, shape=(None, self.goal_dim))

        # Dimensions of action placeholder will differ depending on layer level
        if layer_number == 0:
            action_dim = env.action_dim
        else:
            action_dim = env.subgoal_dim

        self.action_ph = tf.placeholder(tf.float32, shape=(None, action_dim), name='action_ph')
        
        self.features_ph = tf.concat([self.state_ph, self.goal_ph, self.action_ph], axis=1)

        # Set parameters to give critic optimistic initialization near q_init
        self.q_init = -0.067
        self.q_offset = -np.log(self.q_limit/self.q_init - 1)

        # Create critic network graph
        self.infer = self.create_nn(self.features_ph)
        self.weights = [v for v in tf.trainable_variables() if self.critic_name in v.op.name]

        # Create target critic network graph.  Please note that by default the critic networks are not used and updated.  To use critic networks please follow instructions in the "update" method in this file and the "learn" method in the "layer.py" file.

        # Target network code "repurposed" from Patrick Emani :^)
        self.target = self.create_nn(self.features_ph, name = self.critic_name + '_target')
        self.target_weights = [v for v in tf.trainable_variables() if self.critic_name in v.op.name][len(self.weights):]

        self.update_target_weights = \
	    [self.target_weights[i].assign(tf.multiply(self.weights[i], self.tau) +
                                                  tf.multiply(self.target_weights[i], 1. - self.tau))
                    for i in range(len(self.target_weights))]
	
        self.wanted_qs = tf.placeholder(tf.float32, shape=(None, 1))

        self.loss = tf.reduce_mean(tf.square(self.wanted_qs - self.infer))

        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.gradient = tf.gradients(self.infer, self.action_ph)


    def get_Q_value(self,state, goal, action):
        return self.sess.run(self.infer,
                feed_dict={
                    self.state_ph: state,
                    self.goal_ph: goal,
                    self.action_ph: action
                })[0]

    def get_target_Q_value(self,state, goal, action):
        return self.sess.run(self.target,
                feed_dict={
                    self.state_ph: state,
                    self.goal_ph: goal,
                    self.action_ph: action
                })[0]


    def update(self, old_states, old_actions, rewards, new_states, goals, new_actions, is_terminals):

        # Be default, repo does not use target networks.  To use target networks, comment out "wanted_qs" line directly below and uncomment next "wanted_qs" line.  This will let the Bellman update use Q(next state, action) from target Q network instead of the regular Q network.  Make sure you also make the updates specified in the "learn" method in the "layer.py" file.  
        wanted_qs = self.sess.run(self.infer,
                feed_dict={
                    self.state_ph: new_states,
                    self.goal_ph: goals,
                    self.action_ph: new_actions
                })

        """
        # Uncomment to use target networks
        wanted_qs = self.sess.run(self.target,
                feed_dict={
                    self.state_ph: new_states,
                    self.goal_ph: goals,
                    self.action_ph: new_actions
                })
        """
       
        for i in range(len(wanted_qs)):
            if is_terminals[i]:
                wanted_qs[i] = rewards[i]
            else:
                wanted_qs[i] = rewards[i] + self.gamma * wanted_qs[i][0]

            # Ensure Q target is within bounds [-self.time_limit,0]
            wanted_qs[i] = max(min(wanted_qs[i],0), self.q_limit)
            assert wanted_qs[i] <= 0 and wanted_qs[i] >= self.q_limit, "Q-Value target not within proper bounds"


        self.loss_val, _ = self.sess.run([self.loss, self.train],
                feed_dict={
                    self.state_ph: old_states,
                    self.goal_ph: goals,
                    self.action_ph: old_actions,
                    self.wanted_qs: wanted_qs 
                })

        

    def get_gradients(self, state, goal, action):
        grads = self.sess.run(self.gradient,
                feed_dict={
                    self.state_ph: state,
                    self.goal_ph: goal,
                    self.action_ph: action
                })

        return grads[0]

    # Function creates the graph for the critic function.  The output uses a sigmoid, which bounds the Q-values to between [-Policy Length, 0].
    def create_nn(self, features, name=None):

        if name is None:
            name = self.critic_name        

        with tf.variable_scope(name + '_fc_1'):
            fc1 = layer(features, 64)
        with tf.variable_scope(name + '_fc_2'):
            fc2 = layer(fc1, 64)
        with tf.variable_scope(name + '_fc_3'):
            fc3 = layer(fc2, 64)
        with tf.variable_scope(name + '_fc_4'):
            fc4 = layer(fc3, 1, is_output=True)

            # A q_offset is used to give the critic function an optimistic initialization near 0
            output = tf.sigmoid(fc4 + self.q_offset) * self.q_limit

        return output
