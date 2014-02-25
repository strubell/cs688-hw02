'''
CS688 HW02: Chain CRF

Question 2: Sum-Product Message Passing

@author: Emma Strubell
'''

import chain_crf as crf
from data_loader import *
import itertools

'''
Compute clique potentials for the first test word
'''
def q2_1(): 
    data_fname = "../data/test_img1.txt"
    data = load_data(data_fname)
    f_weights = load_feature_weights()
    t_weights = load_transition_weights()
    node_potentials = crf.compute_node_potentials(data, f_weights)
    clique_potentials = crf.compute_clique_potentials(node_potentials, t_weights)
    indices = (0,1,8) # only want e, t, r
    print clique_potentials[...,indices][:,indices]

'''
Compute log-space sum-product messages for the first test word
'''
def q2_2(): 
    data_fname = "../data/test_img1.txt"
    data = load_data(data_fname)
    f_weights = load_feature_weights()
    t_weights = load_transition_weights()
    node_potentials = crf.compute_node_potentials(data, f_weights)
    clique_potentials = crf.compute_clique_potentials(node_potentials, t_weights)
    forward, backward = crf.compute_messages(clique_potentials)
    print "forward:\n", forward
    print "backward:\n", backward

'''
Compute log beliefs at each node in the clique tree for the first test word
'''
def q2_3(): 
    data_fname = "../data/test_img1.txt"
    data = load_data(data_fname)
    f_weights = load_feature_weights()
    t_weights = load_transition_weights()
    node_potentials = crf.compute_node_potentials(data, f_weights)
    clique_potentials = crf.compute_clique_potentials(node_potentials, t_weights)
    forward, backward = crf.compute_messages(clique_potentials)
    beliefs = crf.compute_beliefs(clique_potentials, forward, backward)
    print "beliefs:\n", beliefs[:,:2,:2] # only want e, t
    
'''
Compute marginals, pairwise marginals
'''
def q2_4(): 
    data_fname = "../data/test_img1.txt"
    data = load_data(data_fname)
    f_weights = load_feature_weights()
    t_weights = load_transition_weights()
    node_potentials = crf.compute_node_potentials(data, f_weights)
    clique_potentials = crf.compute_clique_potentials(node_potentials, t_weights)
    forward, backward = crf.compute_messages(clique_potentials)
    beliefs = crf.compute_beliefs(clique_potentials, forward, backward)
    pos_probs, trans_probs = crf.compute_marginals(clique_potentials, beliefs)
    print "marginals:\n", pos_probs
    print "pairwise:\n", trans_probs

'''

'''
def q2_5(): 
    data_fname = "../data/test_img1.txt"
    data = load_data(data_fname)
    f_weights = load_feature_weights()
    t_weights = load_transition_weights()
    node_potentials = crf.compute_node_potentials(data, f_weights)
    clique_potentials = crf.compute_clique_potentials(node_potentials, t_weights)
    forward, backward = crf.compute_messages(clique_potentials)
    beliefs = crf.compute_beliefs(clique_potentials, forward, backward)
    pos_probs, trans_probs = crf.compute_marginals(clique_potentials, beliefs)
    print crf.classify(pos_probs, trans_probs)

print "Question 2.1:"
q2_1()

print "Question 2.2:"
q2_2()

print "Question 2.3:"
q2_3()

print "Question 2.4:"
q2_4()

print "Question 2.5:"
q2_5()