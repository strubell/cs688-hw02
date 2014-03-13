'''
CS688 HW02: Chain CRF

Question 5: CRF Training

@author: Emma Strubell
'''
from __future__ import division
from data_loader import *
import chain_crf as crf
from scipy.optimize import minimize
import numpy as np

tolerance = 1e-10
train_size = 50
gold_words = load_gold('train')
training_data = [load_data("../data/train_img%d.txt" % (i+1)) for i in range(train_size)]
num_labels = 10
num_features = 321
num_f_weights = num_labels*num_features 

def get_sum_prod_marginals(num_examples, f_weights, t_weights):
    all_node_potentials = [crf.compute_node_potentials(example, f_weights) for example in training_data[:num_examples]]
    all_clique_potentials = [crf.compute_clique_potentials(node_potentials, t_weights) for node_potentials in all_node_potentials]
    all_fb = [crf.compute_messages(clique_potentials) for clique_potentials in all_clique_potentials]
    all_beliefs = [crf.compute_beliefs(all_clique_potentials[i], all_fb[i][0], all_fb[i][1]) for i in range(num_examples)]
    return np.array([crf.compute_marginals(all_clique_potentials[i], all_beliefs[i]) for i in range(num_examples)])

# objective function: average log likelihood
def avg_log_likelihood(x):
    # reshape input to get position and transition parameters
    f_weights = np.reshape(x[:num_f_weights], (num_labels, num_features))
    t_weights = np.reshape(x[num_f_weights:], (num_labels, num_labels))
    all_pos_probs = get_sum_prod_marginals(train_size, f_weights, t_weights)[:,0]
    return -np.mean([np.sum(np.log([all_pos_probs[i][j,k] for j,k in enumerate(gold_words[i])])) for i in range(train_size)])

# gradient of objective function
def gradient(x):
    # reshape input to get position and transition parameters
    f_weights = np.reshape(x[:num_f_weights], (num_labels, num_features))
    t_weights = np.reshape(x[num_f_weights:], (num_labels, num_labels))
    all_marginals = get_sum_prod_marginals(train_size, f_weights, t_weights)
    all_pos_probs = all_marginals[:,0]
    all_pairwise_probs = all_marginals[:,1]
    f_gradient = np.zeros((num_labels,num_features))
    t_gradient = np.zeros((num_labels,num_labels))
    for idx,example in enumerate(training_data):
        # compute feature gradient
        indicators = np.zeros((len(example),num_labels))
        settings = zip(range(len(example)),gold_words[idx])
        for setting in settings:
            indicators[setting] = 1.0
        f_gradient += np.transpose(np.matrix([indicators[i]-all_pos_probs[idx][i] for i in range(len(example))]))*example
        
        # compute transition gradient
        transitions = [(i, p[0], p[1]) for i,p in enumerate(zip(gold_words[idx],gold_words[idx][1:]))]
        indicators = np.zeros((len(example)-1,num_labels,num_labels))
        for trans in transitions:
            indicators[trans] = 1.0
        t_gradient += np.sum(indicators-all_pairwise_probs[idx],axis=0)
    f_gradient /= train_size
    t_gradient /= train_size
    return -np.hstack((np.ravel(f_gradient),np.ravel(t_gradient)))

# learn feature and transition parameters for CRF
print "Learning parameters for model using %d training instances" % (train_size)
x0 = np.zeros((num_labels*num_labels)+(num_labels*num_features))
result = minimize(avg_log_likelihood, x0, jac=gradient, method='L-BFGS-B')

# use learned parameters to predict labels
f_weights = np.reshape(result['x'][:num_f_weights], (num_labels, num_features))
t_weights = np.reshape(result['x'][num_f_weights:], (num_labels, num_labels))

print "Testing..."
gold_words = load_gold('test')
test_size = len(gold_words)
test_chars_size = 0
word_acc = 0.0
char_acc = 0.0
for i in range(1, test_size + 1):
    data_fname = "../data/test_img%d.txt" % (i)
    data = load_data(data_fname)
    node_potentials = crf.compute_node_potentials(data, f_weights)
    clique_potentials = crf.compute_clique_potentials(node_potentials, t_weights)
    forward, backward = crf.compute_messages(clique_potentials)
    beliefs = crf.compute_beliefs(clique_potentials, forward, backward)
    pos_probs, _ = crf.compute_marginals(clique_potentials, beliefs)
    word = crf.classify(pos_probs)
    gold = ''.join(map(lambda c: char_map[c], gold_words[i - 1]))
    word_acc += 1.0 if word == gold else 0.0
    chars_correct = sum([1 if word[j] == gold[j] else 0 for j in range(len(word))])
    char_acc += chars_correct
    test_chars_size += len(word)
word_acc /= test_size
char_acc /= test_chars_size
print "Character accuracy:", char_acc
