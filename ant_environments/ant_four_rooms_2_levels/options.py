import argparse

"""
Below are training options user can specify in command line.

Options Include:

1. Retrain boolean ("--retrain")
- If included, actor and critic neural network parameters are reset

2. Testing boolean ("--test")
- If included, agent only uses greedy policy without noise.  No changes are made to policy and neural networks.
- If not included, periods of training are by default interleaved with periods of testing to evaluate progress.

3. Show boolean ("--show")
- If included, training will be visualized

4. Train Only boolean ("--train_only")
- If included, agent will be solely in training mode and will not interleave periods of training and testing

5. Verbosity boolean ("--verbose")
- If included, summary of each transition will be printed

6. All Trans boolean ("--all_trans")
- If included, all transitions including (i) hindsight action, (ii) subgoal penalty, (iii) preliminary HER, and (iv) final HER transitions will be printed.  Use below options to print out specific types of transitions.

7. Hindsight Action trans boolean ("hind_action")
- If included, prints hindsight actions transitions for each level

8. Subgoal Penalty trans ("penalty")
- If included, prints the subgoal penalty transitions

9. Preliminary HER trans ("prelim_HER")
-If included, prints the preliminary HER transitions (i.e., with TBD reward and goal components)

10.  HER trans ("HER")
- If included, prints the final HER transitions for each level

11. Show Q-values ("--Q_values")
- Show Q-values for each action by each level

"""

def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Include to reset policy'
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
        '--train_only',
        action='store_true',
        help='Include to use training mode only'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print summary of each transition'
    )

    parser.add_argument(
        '--all_trans',
        action='store_true',
        help='Print summary of each transition'
    )

    parser.add_argument(
        '--hind_action',
        action='store_true',
        help='Print summary of each transition'
    )

    parser.add_argument(
        '--penalty',
        action='store_true',
        help='Print summary of each transition'
    )

    parser.add_argument(
        '--prelim_HER',
        action='store_true',
        help='Print summary of each transition'
    )

    parser.add_argument(
        '--HER',
        action='store_true',
        help='Print summary of each transition'
    )

    parser.add_argument(
        '--Q_values',
        action='store_true',
        help='Print summary of each transition'
    )

    FLAGS, unparsed = parser.parse_known_args()


    return FLAGS
