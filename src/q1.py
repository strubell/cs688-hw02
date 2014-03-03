'''
CS688 HW02: Chain CRF

Question 1: Exhaustive Inference

@author: Emma Strubell
'''

import chain_crf as crf
from data_loader import *
import itertools

'''
Compute feature potentials for first test word
'''
def q1_1(): 
    data_fname = "../data/test_img1.txt"
    data = load_data(data_fname)
    f_weights = load_feature_weights()
    potentials = crf.compute_node_potentials(data, f_weights)
    print potentials
    # uncomment to print LaTeX tabular version
    #print_marginals_table(potentials)

'''
Compute negative energy of true label sequence for first three test words
'''
def q1_2():
    data_fnames = ["../data/test_img1.txt", \
                   "../data/test_img2.txt", \
                   "../data/test_img3.txt"] 
    t_weights = load_transition_weights()
    f_weights = load_feature_weights()
    gold_vals = load_gold('test')
    for i,fname in enumerate(data_fnames):
        data = load_data(fname)
        potentials = crf.compute_node_potentials(data, f_weights)
        print crf.compute_neg_energy(potentials, t_weights, gold_vals[i])
    
'''
Compute log partition function by exhaustive summation for the first three
test words.
'''    
def q1_3():
    data_fnames = ["../data/test_img1.txt", \
                   "../data/test_img2.txt", \
                   "../data/test_img3.txt"] 
    t_weights = load_transition_weights()
    f_weights = load_feature_weights()
    for fname in data_fnames:
        data = load_data(fname)
        potentials = crf.compute_node_potentials(data, f_weights)
        seqs = itertools.product('0123456789', repeat=np.shape(potentials)[0])
        energies = [crf.compute_neg_energy(potentials, t_weights, seq) for seq in seqs]
        print crf.log_Z(energies)

'''
Compute most likely joint labeling for first three test words.
'''    
def q1_4():
    data_fnames = ["../data/test_img1.txt", \
                   "../data/test_img2.txt", \
                   "../data/test_img3.txt"] 
    t_weights = load_transition_weights()
    f_weights = load_feature_weights()
    for fname in data_fnames:
        data = load_data(fname)
        potentials = crf.compute_node_potentials(data, f_weights)
        print crf.infer_labels(potentials, t_weights)

'''
Compute marginal probability distribution over character labels for
each position in first test word.
'''    
def q1_5():
    data_fname = "../data/test_img1.txt"
    data = load_data(data_fname)
    f_weights = load_feature_weights()
    t_weights = load_transition_weights()
    gold_seq = load_gold('test')[0]
    marginals = crf.compute_exhaustive_marginals(data, f_weights, t_weights, gold_seq)
    print marginals
    # uncomment to print LaTeX tabular version
    #print_marginals_table(marginals)

print "Question 1.1:"
q1_1()

print "Question 1.2:"
q1_2()

print "Question 1.3:"
#q1_3()

print "Question 1.4:"
#q1_4()

print "Question 1.5:"
q1_5()