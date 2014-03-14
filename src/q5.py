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
import time
import matplotlib.pyplot as plt

tolerance = 1e-10
total_train_size = 400
gold_train_words = load_gold('train')
gold_test_words = load_gold('test')
test_size = len(gold_test_words)
training_data = [load_data("../data/train_img%d.txt" % (i+1)) for i in range(total_train_size)]
test_data = [load_data("../data/test_img%d.txt" % (i)) for i in range(1, test_size + 1)]
num_labels = 10
num_features = 321
num_f_weights = num_labels*num_features 

# call sum-product message passing code and get pairwise and position marginals
def get_sum_prod_marginals(num_examples, f_weights, t_weights):
    all_node_potentials = [crf.compute_node_potentials(example, f_weights) for example in training_data[:num_examples]]
    all_clique_potentials = [crf.compute_clique_potentials(node_potentials, t_weights) for node_potentials in all_node_potentials]
    all_fb = [crf.compute_messages(clique_potentials) for clique_potentials in all_clique_potentials]
    all_beliefs = [crf.compute_beliefs(all_clique_potentials[i], all_fb[i][0], all_fb[i][1]) for i in range(num_examples)]
    return np.array([crf.compute_marginals(all_clique_potentials[i], all_beliefs[i]) for i in range(num_examples)])

# objective function: average log likelihood
def avg_log_likelihood(x, train_size):
    # reshape input to get position and transition parameters
    f_weights = np.reshape(x[:num_f_weights], (num_labels, num_features))
    t_weights = np.reshape(x[num_f_weights:], (num_labels, num_labels))
    all_pos_probs = get_sum_prod_marginals(train_size, f_weights, t_weights)[:,0]
    return -np.mean([np.sum(np.log([all_pos_probs[i][j,k] for j,k in enumerate(gold_train_words[i])])) for i in range(train_size)])

# gradient of objective function with respect to feature and transition weights
def gradient(x, train_size):
    # reshape input to get position and transition parameters
    f_weights = np.reshape(x[:num_f_weights], (num_labels, num_features))
    t_weights = np.reshape(x[num_f_weights:], (num_labels, num_labels))
    all_marginals = get_sum_prod_marginals(train_size, f_weights, t_weights)
    all_pos_probs = all_marginals[:,0]
    all_pairwise_probs = all_marginals[:,1]
    f_gradient = np.zeros((num_labels,num_features))
    t_gradient = np.zeros((num_labels,num_labels))
    for idx,example in enumerate(training_data[:train_size]):
        # compute feature gradient
        indicators = np.zeros((len(example),num_labels))
        settings = zip(range(len(example)),gold_train_words[idx])
        for setting in settings:
            indicators[setting] = 1.0
        f_gradient += np.transpose(np.matrix([indicators[i]-all_pos_probs[idx][i] for i in range(len(example))]))*example
        
        # compute transition gradient
        transitions = [(i, p[0], p[1]) for i,p in enumerate(zip(gold_train_words[idx],gold_train_words[idx][1:]))]
        indicators = np.zeros((len(example)-1,num_labels,num_labels))
        for trans in transitions:
            indicators[trans] = 1.0
        t_gradient += np.sum(indicators-all_pairwise_probs[idx],axis=0)
    f_gradient /= train_size
    t_gradient /= train_size
    return -np.hstack((np.ravel(f_gradient),np.ravel(t_gradient)))

def test_model(f_weights, t_weights):
    test_chars_size = 0
    word_acc = 0.0
    char_acc = 0.0
    for i in range(test_size):
        node_potentials = crf.compute_node_potentials(test_data[i], f_weights)
        clique_potentials = crf.compute_clique_potentials(node_potentials, t_weights)
        forward, backward = crf.compute_messages(clique_potentials)
        beliefs = crf.compute_beliefs(clique_potentials, forward, backward)
        pos_probs, _ = crf.compute_marginals(clique_potentials, beliefs)
        word = crf.classify(pos_probs)
        gold = ''.join(map(lambda c: char_map[c], gold_test_words[i]))
        word_acc += 1.0 if word == gold else 0.0
        chars_correct = sum([1 if word[j] == gold[j] else 0 for j in range(len(word))])
        char_acc += chars_correct
        test_chars_size += len(word)
    word_acc /= test_size
    char_acc /= test_chars_size
    return 1-char_acc

def train_model(train_size):
    # learn feature and transition parameters for CRF
    print "Learning parameters for model using %d training instances..." % (train_size)
    x0 = np.zeros((num_labels*num_labels)+(num_labels*num_features))
    
    t0 = time.time()
    result = minimize(avg_log_likelihood, x0, jac=gradient, method='L-BFGS-B', args=[train_size])
    f_weights = np.reshape(result['x'][:num_f_weights], (num_labels, num_features))
    t_weights = np.reshape(result['x'][num_f_weights:], (num_labels, num_labels))
    t1 = time.time()-t0
    
    ll = avg_log_likelihood(result['x'], train_size)
    return f_weights, t_weights, t1, ll

# train and test 8 models on 50, 100, 150, 200, 250, 300, 350 and 400 training examples
num_examples = [50, 100, 150, 200, 250, 300, 350, 400]
models = [train_model(i) for i in num_examples]
errors = [test_model(m[0], m[1]) for m in models]
speeds = [m[2] for m in models]
log_likelihoods = [-m[3] for m in models]
print "Train sizes:", num_examples
print "Speeds:", speeds
print "Errors:", errors
print "Log likelihoods:", log_likelihoods

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_title("Training Speed vs. Number of Training Examples")
ax1.set_xlabel("Number of Training Examples")
ax1.set_ylabel("Speed (seconds)")
ax1.plot(num_examples, speeds, '-o')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.set_title("Test Error vs. Number of Training Examples")
ax2.set_xlabel("Number of Training Examples")
ax2.set_ylabel("Error")
ax2.plot(num_examples, errors, '-o')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.set_title("Average Log Likelihood vs. Number of Training Examples")
ax3.set_xlabel("Number of Training Examples")
ax3.set_ylabel("Average Log Likelihood")
ax3.plot(num_examples, log_likelihoods, '-o')

plt.show()
