'''
CS688 HW02: Chain CRF

Question 3: Maximum Log Likelihood Learning Derivation

@author: Emma Strubell
'''

from data_loader import *
import chain_crf as crf

# compute log likelihood over first 50 training examples
gold_words = load_gold('train')
train_size = 50
f_weights = load_feature_weights()
t_weights = load_transition_weights()
total = 0.0
for i in range(train_size):
    data_fname = "../data/train_img%d.txt" % (i+1)
    example = load_data(data_fname)
    node_potentials = crf.compute_node_potentials(example, f_weights)
    clique_potentials = crf.compute_clique_potentials(node_potentials, t_weights)
    forward, backward = crf.compute_messages(clique_potentials)
    beliefs = crf.compute_beliefs(clique_potentials, forward, backward)
    pos_probs, _ = crf.compute_marginals(clique_potentials, beliefs) 
    total += np.sum(np.log([pos_probs[i,j] for i,j in enumerate(gold_words[i])]))
total /= train_size
print "Log likelihood:", total