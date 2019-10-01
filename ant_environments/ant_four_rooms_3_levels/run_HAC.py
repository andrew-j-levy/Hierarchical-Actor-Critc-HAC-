"""
"run_HAC.py" executes the training schedule for the agent.  By default, the agent will alternate between exploration and testing phases.  The number of episodes in the exploration phase can be configured in section 3 of "design_agent_and_env.py" file.  If the user prefers to only explore or only test, the user can enter the command-line options ""--train_only" or "--test", respectively.  The full list of command-line options is available in the "options.py" file.
"""

import pickle as cpickle
import agent as Agent
from utils import print_summary

NUM_BATCH = 1000
TEST_FREQ = 2

num_test_episodes = 100

def run_HAC(FLAGS,env,agent):

    # Print task summary
    print_summary(FLAGS,env)

    total_episodes = 0

    # Determine training mode.  If not testing and not solely training, interleave training and testing to track progress
    mix_train_test = False
    if not FLAGS.test and not FLAGS.train_only:
        mix_train_test = True

    for batch in range(NUM_BATCH):

        num_episodes = agent.other_params["num_exploration_episodes"]

        # Evaluate policy every TEST_FREQ batches if interleaving training and testing
        if mix_train_test and batch % TEST_FREQ == 0:
            print("\n--- TESTING ---")
            agent.FLAGS.test = True
            num_episodes = num_test_episodes

            # Reset successful episode counter
            successful_episodes = 0

        for episode in range(num_episodes):

            print("\nBatch %d, Episode %d" % (batch, episode))

            # Train for an episode
            success = agent.train(env, episode, total_episodes)

            if success:
                print("Batch %d, Episode %d End Goal Achieved\n" % (batch, episode))

                # Increment successful episode counter if applicable
                if mix_train_test and batch % TEST_FREQ == 0:
                    successful_episodes += 1

            if FLAGS.train_only or (mix_train_test and batch % TEST_FREQ != 0):
                total_episodes += 1

        # Save agent
        agent.save_model(episode)

        # Finish evaluating policy if tested prior batch
        if mix_train_test and batch % TEST_FREQ == 0:

            # Log performance
            success_rate = successful_episodes / num_test_episodes * 100
            print("\nTesting Success Rate %.2f%%" % success_rate)
            agent.log_performance(success_rate)
            agent.FLAGS.test = False

            print("\n--- END TESTING ---\n")
