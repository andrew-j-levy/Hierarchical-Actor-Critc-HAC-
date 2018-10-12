"""
This is the starting file for the Hierarchical Actor-Critc (HAC) algorithm.  The below script processes the command-line options specified by the user and instantiates the environment and agent. 
"""

from design_agent_and_env import design_agent_and_env
from options import parse_options
from agent import Agent
from run_HAC import run_HAC

# Determine training options specified by user.  The full list of available options can be found in "options.py" file.
FLAGS = parse_options()

# Instantiate the Mujoco environment.  The designer must provide the information requested in the "design_env.py" file. 
agent, env = design_agent_and_env(FLAGS)

# Begin training
run_HAC(FLAGS,env,agent)
