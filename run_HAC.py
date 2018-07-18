# Train agent for the number of episodes and in the mode specified by the user.

import pickle as cpickle
import agent as Agent
from utils import print_summary

NUM_BATCH = 1000
NUM_EPISODES = 100
TEST_FREQ = 2

def run_HAC(FLAGS,env,agent):

    # Print task summary
    print_summary(FLAGS,env)
    
    # Determine training mode.  If not testing and not solely training, interleave training and testing to track progress
    mix_train_test = False
    if not FLAGS.test and not FLAGS.train_only:
        mix_train_test = True
     
    for batch in range(NUM_BATCH):
        
        # Evaluate policy every TEST_FREQ batches if interleaving training and testing
        if mix_train_test and batch % TEST_FREQ == 0:
            print("\n--- TESTING ---")
            agent.FLAGS.test = True
            
            # Reset successful episode counter
            successful_episodes = 0

        for episode in range(NUM_EPISODES):
            
            print("\nBatch %d, Episode %d" % (batch, episode))
            success = agent.train(env, episode)

            if success:
                print("Batch %d, Episode %d End Goal Achieved\n" % (batch, episode))
                
                # Increment successful episode counter if applicable
                if mix_train_test and batch % TEST_FREQ == 0:
                    successful_episodes += 1            

        # Save agent
        agent.save_model(episode)
           
        # Finish evaluating policy if tested prior batch
        if mix_train_test and batch % TEST_FREQ == 0:

            # Log performance
            success_rate = successful_episodes / NUM_EPISODES * 100
            print("\nTesting Success Rate %.2f%%" % success_rate)
            agent.log_performance(success_rate)
            agent.FLAGS.test = False

            print("\n--- END TESTING ---\n")

            

    
    

     
