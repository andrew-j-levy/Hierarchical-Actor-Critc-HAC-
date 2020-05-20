# Hierarchical Actor-Critc (HAC)
This repository contains the code to implement the *Hierarchical Actor-Critic (HAC)* algorithm.  HAC helps agents learn tasks more quickly by enabling them to break problems down into short sequences of actions.  For more information on the algorithm, please see our ICLR 2019 [paper](https://openreview.net/pdf?id=ryzECoAcY7) and [blog post](http://bigai.cs.brown.edu/2019/09/03/hac.html).

To run HAC, execute the command *"python3 initialize_HAC.py --retrain"*.  By default, this will train a UR5 agent with a 3-level hierarchy to learn to achieve certain poses.  This UR5 agent should achieve a 90+% success rate in around 350 episodes.  The following [video](https://www.youtube.com/watch?v=R86Vs9Vb6Bc) shows how a 3-layered agent performed after 450 episodes of training.  In order to watch your trained agent, execute the command *"python3 initialize_HAC.py --test --show"*.  To train agents in the inverted pendulum domain, swap the UR5 reacher *"design_agent_and_env.py"* file for an inverted pendulum *"design_agent_and_env.py"* file, which are located in *"example_designs"* folder folder.  To train agents in the ant reacher and ant four rooms environments, execute the command *"python3 initialize_HAC.py --retrain"* in the appropriate folder within the *ant_environments* directory.  In the near future, the code for the ant domains will be integrated with the code for the other domains.

Please note that in order to run this repository, you must have (i) a MuJoCo [license](https://www.roboti.us/license.html), (ii) the required MuJoCo software [libraries](https://www.roboti.us/index.html), and (iii) the MuJoCo Python [wrapper](https://github.com/openai/mujoco-py) from OpenAI.  

Happy to answer any questions you have.  Please email me at andrew_levy2@brown.edu.

## UPDATE LOG

### 5/20/2020 - Key Changes

1. Added 2-level ant environments
2. Centralized exploration hyperparameters for ant environments in *design_agent_and_env.py* file

### 2/25/2020 - Key Changes

1. TensorFlow 2.x Compatible

2. Fine-tuned exploration parameters of the Ant Reacher environment

### 10/1/2019 - Key Changes

1.  Added Ant Reacher and Ant Four Rooms Environments

The code for the ant environments has been temporaily added to the *ant_environments* folder.  In the near future, the code for the ant domains will be integrated with the code for the other domains.  Only minimal changes to the code are needed to run the ant environments.

### 10/12/2018 - Key Changes
1.  Bounded Q-Values

The Q-values output by the critic network at each level are now bounded between *[-T,0]*, in which *T* is the max sequence length in which each policy specializes as well as the negative of the subgoal penalty.  We use an upper bound of 0 because our code uses a nonpositive reward function.  Consequently, Q-values should never be positive.  However, we noticed that somtimes the critic function approximator would make small mistakes and assign positive Q-values, which occassionally proved harmful to results.  In addition, we observed improved results when we used a tighter lower bound of *-T* (i.e., the subgoal penalty).  The improved results may result from the increased flexibility the bounded Q-values provides the critic.  The critic can assign a value of *-T* to any (state,action,goal) tuple, in which the action does not bring the agent close to the goal, instead of having to learn the exact value.

2.  Removed Target Networks

We also noticed improved results when we used the regular Q-networks to determine the Bellman target updates (i.e., *reward + Q(next state,pi(next state),goal)*) instead of the separate target networks that are used in DDPG.  The default setting of our code base thus no longer uses target networks.  However, the target networks can be easily activated by making the changes specified in (i) the *"learn"* method in the *"layer.py"* file and (ii) the *"update"* method in the *"critic.py"* file.  

3.  Centralized Design Template

Users can now configure the agent and environment in the single file, *"design_agent_and_env.py"*.  This template file contains most of the significant hyperparameters in HAC.  We have removed the command-line options that can change the architecture of the agent's hierarchy.

4.  Added UR5 Reacher Environment

We have added a new UR5 reacher environment, in which a UR5 agent can learn to achieve various poses.  The *"ur5.xml"* MuJoCo file also contains commented code for a Robotiq gripper if you would like to augment the agent.  Additional environments will hopefully be added shortly.  
