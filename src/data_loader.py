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

def load_data(fname):
    return np.loadtxt(fname)

def load_feature_weights():
    return np.loadtxt(feature_weights_fname)

def load_transition_weights():
    return np.loadtxt(transition_weights_fname)

def load_gold(gold_type):
    gold_file = open(gold_test_fname if gold_type == 'test' else gold_train_fname)
    return [map(lambda c: int_map[c], word[:-1]) for word in gold_file.readlines()]