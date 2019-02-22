#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

# activation function: sigmoid, softmax(output layer), tanh, rectified linear
# cost function: quadratic, cross-entropy
# regularization: none, L2, L1, dropout, artificially expanding training data
# weight initialization: all zero, random, random with good variance
# stochastic gradient descent: momentum-based

# hyper-parameters: learning rate, epoch (early stopping), regularization lambda, mini-batch size

class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.init_weights_biases()
        
    def init_weights_biases(self):
        self.weights = np.random.randn()
        self.biases = 0
        
    