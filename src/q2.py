'''
CS688 HW02: Chain CRF

Question 2: Sum-Product Message Passing

@author: Emma Strubell
'''
from __future__ import division
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
    # uncomment to print LaTeX tabular version
    #print_marginals_table(pos_probs)
    indices = (0,1,8) # only want e, t, r
    print "pairwise:\n", trans_probs[...,indices][:,indices]

'''
Classify each word in the test set, report accuracy and max probability words
'''
def q2_5(): 
    gold_words = load_gold('test')
    test_size = len(gold_words)
    test_chars_size = 0
    word_acc = 0.0
    char_acc = 0.0
    for i in range(1,test_size+1):
        data_fname = "../data/test_img%d.txt" % (i)
        data = load_data(data_fname)
        f_weights = load_feature_weights()
        t_weights = load_transition_weights()
        node_potentials = crf.compute_node_potentials(data, f_weights)
        clique_potentials = crf.compute_clique_potentials(node_potentials, t_weights)
        forward, backward = crf.compute_messages(clique_potentials)
        beliefs = crf.compute_beliefs(clique_potentials, forward, backward)
        pos_probs, _ = crf.compute_marginals(clique_potentials, beliefs)
        word = crf.classify(pos_probs)
        gold = ''.join(map(lambda c: char_map[c], gold_words[i-1]))
        word_acc += 1.0 if word==gold else 0.0
        chars_correct = sum([1 if word[j]==gold[j] else 0 for j in range(len(word))])
        char_acc += chars_correct
        test_chars_size += len(word)
        if i < 6 : # only print first 5 test words
            print word, gold
    word_acc /= test_size
    char_acc /= test_chars_size
    print "Word accuracy:", word_acc
    print "Character accuracy:", char_acc

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