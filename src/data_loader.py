'''
CS688 HW02: Chain CRF

For loading data, weights.

@author: Emma Strubell
'''

import numpy as np

feature_weights_fname = "../model/feature-params.txt"
transition_weights_fname = "../model/transition-params.txt"

gold_test_fname = "../data/test_words.txt"
gold_train_fname = "../data/train_words.txt"

# maps char categories to their int value
int_map = {'e': 0, \
            't': 1, \
            'a': 2, \
            'i': 3, \
            'n': 4, \
            'o': 5, \
            's': 6, \
            'h': 7, \
            'r': 8, \
            'd': 9}

# maps ints to their char category value
char_map = {0: 'e', \
            1: 't', \
            2: 'a', \
            3: 'i', \
            4: 'n', \
            5: 'o', \
            6: 's', \
            7: 'h', \
            8: 'r', \
            9: 'd'}

# Print out the marginals as a LaTeX tabular
def print_marginals_table(marginals):
    m_shape = np.shape(marginals)
    print m_shape
    num_pos = m_shape[0]
    marginals = np.transpose(marginals)
    print "\\begin{tabular}{%s}" % (''.join(['r' for i in range(num_pos+1)]))
    print "%s \\\\" % (''.join([" & $x_"+str(i)+"$" for i in range(num_pos)]))
    for i in range(m_shape[1]):
        print "%s %s \\\\" % (char_map[i], ''.join([" & %.6g" % (x) for x in marginals[i]]))
    print "\\end{tabular}"

# Load in a data file
def load_data(fname):
    return np.loadtxt(fname)

# Load the learned feature weights
def load_feature_weights():
    return np.loadtxt(feature_weights_fname)

# Load the learned transition weights
def load_transition_weights():
    return np.loadtxt(transition_weights_fname)

# Load the gold labels of the given type ('train' or 'test')
def load_gold(gold_type):
    gold_file = open(gold_test_fname if gold_type == 'test' else gold_train_fname)
    return [map(lambda c: int_map[c], word[:-1]) for word in gold_file.readlines()]