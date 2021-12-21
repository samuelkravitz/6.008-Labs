#!/usr/bin/env python
# inference.py
# Base code by George H. Chen (georgehc@mit.edu)
# updated 10/05/2021 by Gary Lee (Fall 2021)
#
# Modified by: <your name here!>


# use this to enable/disable graphics
enable_graphics = True

import sys
import numpy as np
import robot
if enable_graphics:
    import graphics


#-----------------------------------------------------------------------------
# Functions for you to implement
#

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states

    all_possible_observed_states: a list of possible observed states

    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state

    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state

    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    robot.py and see how it is used in both robot.py and the function
    generate_data() above, and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    marginals = [None] * num_time_steps
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    forward_messages[0] = robot.Distribution({})
    for state in all_possible_hidden_states:
        if observations[0] is None:
            obs = 1
        else:
            obs = observation_model(state)[observations[0]]
        if observation_model(state)[observations[0]] != 0:
            forward_messages[0][state] = obs * prior_distribution[state]
    forward_messages[0].renormalize()

    for i in range(1, num_time_steps):
        forward_messages[i] = robot.Distribution({})
        for possible_state in all_possible_hidden_states:
            if observations[i] is None:
                obs = 1
            else:
                obs = observation_model(possible_state)[observations[i]]
            total = 0
            for state in forward_messages[i - 1]:
                total += forward_messages[i - 1][state] * transition_model(state)[possible_state]
            if total != 0 and obs != 0:
                forward_messages[i][possible_state] = obs * total
        forward_messages[i].renormalize()

    backward_messages = [None] * num_time_steps
    backward_messages[num_time_steps - 1] = robot.Distribution({})
    for state in all_possible_hidden_states:
        backward_messages[num_time_steps - 1][state] = 1

    for i in range(num_time_steps - 2, -1, -1):
        backward_messages[i] = robot.Distribution({})
        for current_state in all_possible_hidden_states:
            total = 0
            for next_state in backward_messages[i + 1]:
                if observations[i + 1] is None:
                    obs = 1
                else:
                    obs = observation_model(next_state)[observations[i + 1]]
                total += backward_messages[i + 1][next_state] * obs * transition_model(current_state)[next_state]
            if total != 0:
                backward_messages[i][current_state] = total
        backward_messages[i].renormalize()

    #TODO: Compute the marginals
    for i in range(num_time_steps):
        marginals[i] = robot.Distribution({})
        for state in robot.get_all_hidden_states():
            if (forward_messages[i][state] * backward_messages[i][state]) != 0:
                marginals[i][state] = (forward_messages[i][state] * backward_messages[i][state])
        marginals[i].renormalize()

    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.
    tuple([*observation[x], "stay")
    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: This is for you to implement

    num_time_steps = len(observations)
    back_track = [None] * num_time_steps
    estimated_hidden_states = [None] * num_time_steps
    path = [None] * num_time_steps
    back_track[0] = robot.Distribution({})
    for state in all_possible_hidden_states:
        prior = prior_distribution[state]
        if observations[0] is None:
           obs = 1
        else:
            obs = observation_model(state)[observations[0]]
        if obs != 0:
            back_track[0][state] = np.log(prior) + np.log(obs)
    # recursion steps
    for i in range(1, num_time_steps):
        back_track[i] = robot.Distribution({})
        path[i] = {}
        for current_state in all_possible_hidden_states:
            if observations[i] is None:
                obs = 1
            else:
                obs = observation_model(current_state)[observations[i]]
            max_previous = -sys.maxsize
            for prev_state in back_track[i - 1]:
                if transition_model(prev_state)[current_state] != 0:
                    prev_prob = np.log(transition_model(prev_state)[current_state]) + back_track[i - 1][prev_state]
                    if prev_prob > max_previous:
                        max_previous = prev_prob
                        # remember the transition from which previous state is the most likely
                        path[i][current_state] = prev_state
            if obs != 0:
                back_track[i][current_state] = np.log(obs) + max_previous
    estimated_hidden_states[num_time_steps - 1] = max(back_track[num_time_steps - 1], key=back_track[num_time_steps - 1].get)
    for i in range(num_time_steps - 2, -1, -1):
        estimated_hidden_states[i] = path[i + 1][estimated_hidden_states[i + 1]]

    return estimated_hidden_states

def second_best(all_possible_hidden_states,
                all_possible_observed_states,
                prior_distribution,
                transition_model,
                observation_model,
                observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: This is for you to implement

    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this

    return estimated_hidden_states


#-----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#

def generate_data(initial_distribution, transition_model, observation_model,
                  num_time_steps, make_some_observations_missing=False,
                  random_seed=None):
    # generate samples from a hidden Markov model given an initial
    # distribution, transition model, observation model, and number of time
    # steps, generate samples from the corresponding hidden Markov model
    hidden_states = []
    observations  = []

    # if the random seed is not None, then this makes the randomness
    # deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    initial_state       = initial_distribution().sample()
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state   = hidden_states[-1]
        new_state    = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < .1: # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


#-----------------------------------------------------------------------------
# Main
#

if __name__ == '__main__':
    # flags
    make_some_observations_missing = False
    use_graphics                   = enable_graphics
    need_to_generate_data          = True

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg.startswith('--load='):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = \
            generate_data(robot.initial_distribution,
                          robot.transition_model,
                          robot.observation_model,
                          num_time_steps,
                          make_some_observations_missing)

    all_possible_hidden_states   = robot.get_all_hidden_states()
    all_possible_observed_states = robot.get_all_observed_states()
    prior_distribution           = robot.initial_distribution()

    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 robot.transition_model,
                                 robot.observation_model,
                                 observations)
    print('\n')

    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        top_10_states = sorted(marginals[timestep].items(), key=lambda x: x[-1], reverse=True)[:10]
        print([s for s in top_10_states if s[-1]>0])
    else:
        print('*No marginal computed*')
    print('\n')

    timestep = 0
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        top_10_states = sorted(marginals[timestep].items(), key=lambda x: x[-1], reverse=True)[:10]
        print([s for s in top_10_states if s[-1]>0])
    else:
        print('*No marginal computed*')
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               robot.transition_model,
                               robot.observation_model,
                               observations)
    print('\n')

    if num_time_steps > 10:
        print("Last 10 hidden states in the MAP estimate:")
        for time_step in range(num_time_steps - 10, num_time_steps):
            if estimated_states[time_step] is None:
                print('Missing')
            else:
                print(estimated_states[time_step])
        print('\n')

        print('Finding second-best MAP estimate...')
        estimated_states2 = second_best(all_possible_hidden_states,
                                        all_possible_observed_states,
                                        prior_distribution,
                                        robot.transition_model,
                                        robot.observation_model,
                                        observations)
        print('\n')

        print("Last 10 hidden states in the second-best MAP estimate:")
        for time_step in range(num_time_steps - 10 - 1, num_time_steps):
            if estimated_states2[time_step] is None:
                print('Missing')
            else:
                print(estimated_states2[time_step])
        print('\n')

        difference = 0
        for time_step in range(num_time_steps):
            if estimated_states[time_step] != hidden_states[time_step]:
                difference += 1
        print("Number of differences between MAP estimate and true hidden " + \
              "states:", difference)
        true_prob = robot.sequence_prob(hidden_states,
                                       robot.transition_model, robot.observation_model,
                                       prior_distribution, observations)
        print("True Sequence Prob:", true_prob)
        map_prob = robot.sequence_prob(estimated_states,
                                       robot.transition_model, robot.observation_model,
                                       prior_distribution, observations)
        print("MAP Estimate Prob:", map_prob)


        difference = 0
        for time_step in range(num_time_steps):
            if estimated_states2[time_step] != hidden_states[time_step]:
                difference += 1
        print("Number of differences between second-best MAP estimate and " + \
              "true hidden states:", difference)
        map_prob2 = robot.sequence_prob(estimated_states2,
                                       robot.transition_model, robot.observation_model,
                                       prior_distribution, observations)
        print("Second-best MAP Estimate Prob:", map_prob2)

        difference = 0
        for time_step in range(num_time_steps):
            if estimated_states[time_step] != estimated_states2[time_step]:
                difference += 1
        print("Number of differences between MAP and second-best MAP " + \
              "estimates:", difference)

    # display
    if use_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
