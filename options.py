import argparse

"""
Below are training options user can specify in command line.

Options Include:

1. Number of Agent Layers

2. Time limit of each layer

3. Retrain boolean
- If included, neural network parameters are reset

4. Testing boolean
- If included, agent only uses greedy policy without noise.  No changes are made to policy and neural networks. 
- If not included, periods of training are interleaved with period of testing to evaluate progress.

5. Visualization boolean
- If included, training will be visualized

6. Retrain boolean
- If included, actor/critic neural network parameters will be reset

7. Train Only boolean
- If included, agent will be solely in training mode and will not interleave training and testing

8. Verbosity boolean
- If included, summary of each transition will be printed
"""

def int_greater_than_0(string):
    value = int(string)
    if value < 1:
        msg = "Must be a positive integer"
        raise argparse.ArgumentTypeError(msg)
    else:
        return value

def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-l','--layers',
        default=4,
        type = int_greater_than_0,
        help='Number of layers within agent (Req: Int > 0)'
    )

    parser.add_argument(
        '-t','--time_scale',
        default=7,
        type = int_greater_than_0,
        help='Time scale of each layer\'s actions (Req: Int > 0)'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Include to fix current policy'
    )

    parser.add_argument(
        '--show',
        action='store_true',
        help='Include to visualize training'
    )

    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Include to reset policy'
    )

    parser.add_argument(
        '--train_only',
        action='store_true',
        help='Include to use training mode only'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print summary of each transition'
    )

    """
    parser.add_argument(
        '--num_episodes',
        default=100000,
        type = int_greater_than_0,
        help='Number of episdoes to train agent (Req: Int > 0)'
    )
    """


    FLAGS, unparsed = parser.parse_known_args()


    return FLAGS
