# Hierarchical-Actor-Critc-HAC-
This repository contains the code to implement the Hierarchical Actor-Critic (HAC) algorithm.  A more detailed README is coming soon.

To train a 4-layered agent in an inverted pendulum environment, run "python3 initialize_HAC.py --retrain".  The following video (https://www.youtube.com/watch?v=yXNfI3kWUTo) shows how 4-layered agents performed after 1100 episodes of training.  Command line options to customize the agent's hierarchy (# of levels/time limits) or to visualize training can be found in the "options.py" file along with other options.  To train in a different Mujoco environment, follow the instructions in the "design_env.py" file.  

Happy to answer any questions you have.  Please email me at andrew_levy2@brown.edu.
