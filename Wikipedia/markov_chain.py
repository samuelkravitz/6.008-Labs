
from __future__ import division
import sys
import random
import os 
import networkx as nx
import util
import numpy as np
# import matplotlib.pyplot as plt


def part_a(data_filename):
    conditional_distribution = get_graph_distribution(data_filename)
    sample_list = [128, 256, 512, 1024, 2048, 4096, 8192]
    div = []
    for i in sample_list:
        print(i)
        group_1 = approx_markov_chain_steady_state(conditional_distribution, i, 1000)
        group_2 = approx_markov_chain_steady_state(conditional_distribution, i, 1000)
        div.append(compute_div(group_1, group_2))
    # plt.plot(sample_list, div)
    # plt.show()


def approx_markov_chain_steady_state(conditional_distribution, N_samples, iterations_between_samples):
    """
    Computes the steady-state distribution by simulating running the Markov
    chain. Collects samples at regular intervals and returns the empirical
    distribution of the samples.

    Inputs
    ------
    conditional_distribution : A dictionary in which each key is an state,
                               and each value is a Distribution over other
                               states.

    N_samples : the desired number of samples for the approximate empirical
                distribution

    iterations_between_samples : how many jumps to perform between each collected
                                 sample

    Returns
    -------
    An empirical Distribution over the states that should approximate the
    steady-state distribution.
    """
    empirical_distribution = util.Distribution()
    for state in conditional_distribution:
        empirical_distribution[state] = 0
    current = int(random.random() * len(conditional_distribution))
    keys = [_ for _ in conditional_distribution.keys()]
    current = keys[current]
    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE FOR PART (a)
    for i in range(N_samples):
        if i % 5000 == 0:
            print("working")
        for k in range(iterations_between_samples):
            num = random.random()
            if num > .1:
                next_state = conditional_distribution[current].sample()
                if next_state is not None:
                    current = next_state
            else:
                next_state = int(random.random() * len(conditional_distribution))
                next_state = keys[next_state]
                current = next_state
        empirical_distribution[current] += 1
    # END OF YOUR CODE FOR PART (a)
    # -------------------------------------------------------------------------
    empirical_distribution.normalize()
    return empirical_distribution

def get_graph_distribution(filename):
    G = nx.read_gml(filename)
    d = nx.to_dict_of_dicts(G)
    cond_dist = util.Distribution({k: util.Distribution({k_: v_['weight'] for k_,v_ in v.items()}) for k,v in d.items()})
    return cond_dist

def run_pagerank(data_filename, N_samples, iterations_between_samples):
    """
    Runs the PageRank algorithm, and returns the empirical
    distribution of the samples.

    Inputs
    ------
    data_filename : a file with the weighted directed graph on which to run the Markov Chain

    N_samples : the desired number of samples for the approximate empirical
                distribution

    iterations_between_samples : how many jumps to perform between each collected
                                 sample

    Returns
    -------
    An empirical Distribution over the states that should approximate the
    steady-state distribution.
    """
    conditional_distribution = get_graph_distribution(data_filename)

    steady_state = approx_markov_chain_steady_state(conditional_distribution,
                            N_samples,
                            iterations_between_samples)

    pages = conditional_distribution.keys()
    top = sorted( (((steady_state[page]), page) for page in pages), reverse=True )

    values_to_show = min(20, len(steady_state))
    for i in range(0, values_to_show):
        print("%0.6f: %s" %top[i])
    return steady_state

def compute_div(dic_1, dic_2):
    total = 0
    for i in dic_1:
        if dic_2[i] != 0 and dic_1[i] != 0:
            total += dic_1[i] * np.log2(dic_1[i]/dic_2[i])
    return total

def compute_degree(data_filename):
    conditional_distribution = get_graph_distribution(data_filename)
    print(len(conditional_distribution))
    dic = {}
    keys = conditional_distribution.keys()
    for i in keys:
        dic[i] = 0
    for state in conditional_distribution:
        dic[state] += len(conditional_distribution[state])
        nodes = conditional_distribution[state]
        for next in nodes:
            dic[next] += 1
    return dic




if __name__ == '__main__':
    # if len(sys.argv) != 4:
    #     print("Usage: python markovChain.py <data file> <samples> <iterations between samples>")
    #     sys.exit(1)
    # data_filename = sys.argv[1]
    # N_samples = int(sys.argv[2])
    # iterations_between_samples = int(sys.argv[3])
    page = run_pagerank("stat.gml", 25000, 1000)
    page = {key: value for key, value in sorted(page.items())}
    degree = compute_degree("stat.gml")
    degree = {key: value for key, value in sorted(degree.items())}
    page_vals = list(page.values())
    degree_vals = list(degree.values())
    print(len(page_vals))
    print(len(degree_vals))
    # plt.scatter(page_vals, degree_vals)
    # plt.show()
