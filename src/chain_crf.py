'''
CS688 HW02: Chain CRF

Some functions for performing inference using a chain CRF.

@author: Emma Strubell
'''

import numpy as np
from data_loader import *
import itertools

def logsumexp(v, ax=None):
    m = np.max(v)
    return m + np.log(np.sum(np.exp(v-m),axis=ax))

def compute_node_potentials(word, f_weights):
    #potentials = np.reshape(np.matrix(weights)*np.matrix(np.transpose(data)), (4,10))
    potentials = np.array([np.dot(f_weights, letter) for letter in word])
    return potentials

def compute_neg_energy(node_potentials, t_weights, seq):
    phi_f = np.sum([node_potentials[i,j] for i,j in enumerate(seq)])
    phi_t = np.sum([t_weights[seq[i],seq[i+1]] for i in range(len(seq)-1)])
    return phi_f + phi_t

def log_Z(vals):
    return logsumexp(vals)

def infer_labels(node_potentials, t_weights):
    seq_len, categories = np.shape(node_potentials)
    sequences = itertools.product(range(categories), repeat=seq_len)
    energies = np.empty(len(char_map)**seq_len)
    max_energy = np.finfo(np.float64).min
    max_seq = ''
    for i,seq in enumerate(sequences):
        e = compute_neg_energy(node_potentials, t_weights, seq)
        energies[i] = e
        if e > max_energy:
            max_energy = e
            max_seq = seq
    seq_probability = np.exp(max_energy) / np.sum(np.exp(energies))
    seq = map(lambda c: char_map[c], max_seq)
    return ''.join(seq), seq_probability

def compute_exhaustive_marginals(word, f_weights, t_weights, seq):
    potentials = compute_node_potentials(word, f_weights)
    exp_potentials = np.exp(potentials)
    marginals = np.transpose(np.transpose(exp_potentials)/np.sum(exp_potentials, axis=1))
    return marginals

def compute_clique_potentials(node_potentials, t_weights):
    num_cliques = len(node_potentials)-1    
    cliques = np.array([np.transpose(node_potentials[i] + t_weights 
                                        + (node_potentials[i+1][:,np.newaxis]
                                        if i == num_cliques-1 else 0)
                                     ) for i in range(num_cliques)])
    return cliques

def compute_messages(clique_potentials):
    num_cliques, num_categories, _ = np.shape(clique_potentials)
    num_messages = num_cliques-1
    forward_messages = np.empty((num_messages,num_categories))
    backward_messages = np.empty((num_messages,num_categories))
    # forward: propagate prev message, sum over col (axis=0)
    for i in range(num_messages):
        if i == 0:
            # first element in chain; no incoming message
            forward_messages[i] = logsumexp(clique_potentials[i], ax=0)
        else:
            forward_messages[i] = logsumexp(clique_potentials[i] + forward_messages[i-1][:,np.newaxis], ax=0)
    # backward: propagate prev message, sum over rows (axis=1)
    for i,j in enumerate(range(num_messages, 0, -1)):
        if (j == num_cliques-1):
            # last element in chain; no prev message
            backward_messages[i] = logsumexp(clique_potentials[j], ax=1)
        else:
            backward_messages[i] = logsumexp(clique_potentials[j] + backward_messages[i-1], ax=1)
    return forward_messages, backward_messages

def compute_beliefs(clique_potentials, forward_messages, backward_messages):
    num_cliques = np.shape(clique_potentials)[0]
    num_messages = num_cliques-1
    
    # get incoming forward message for this clique
    # first element in chain has no incoming message
    def forward(idx): return forward_messages[idx-1][:,np.newaxis] if idx > 0 else 0.0
    
    # get incoming backward message for this clique
    # last element in chain has no incoming message
    def backward(idx): return backward_messages[num_messages-i-1] if i < num_messages else 0.0
    
    # combine incoming messages with potential for each clique
    beliefs = np.array([clique_potentials[i] + forward(i) + backward(i) for i in range(num_cliques)])
    return beliefs

def compute_marginals(clique_potentials, beliefs):
    num_beliefs = len(beliefs)
    position_probs = np.empty((num_beliefs+1, 10))
    transition_probs = np.empty(np.shape(beliefs))
    # calculate position and transition probabilities
    for i in range(num_beliefs):
        logz = log_Z(beliefs[i])
        position_probs[i] = logsumexp(beliefs[i] - logz, ax=1)#np.sum(np.exp(beliefs[i] - logz), axis=1)
        transition_probs[i] = np.exp(beliefs[i] - logz)
    # sum last position over other variable
    position_probs[-1] = logsumexp(beliefs[-1]-log_Z(beliefs[-1]), ax=0)
    return position_probs, transition_probs

def classify(position_probs, transition_probs):
    print [np.exp(np.max(p)) for p in position_probs]
    return ''.join(map(lambda c: char_map[c], [np.where(p==np.max(p))[0][0] for p in position_probs]))
