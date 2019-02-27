#!/usr/bin/python
# -*- coding: utf-8 -*-

import mnist_loader
from network5 import *

class TestNetwork(object):
    def __init__(self):
        self.training_data, self.validation_data, self.test_data = mnist_loader.load_data_wrapper()
        self.net = Network([784, 30, 10])

    def run(self, epoches = 30, mini_batch_size = 10, learning_rate = 3.0):
        self.net.sgd(self.training_data, epoches, mini_batch_size, learning_rate, test_data = self.test_data)

    def run_test(self, test, init_weight_func = WeightRandom()):
        self.net.init_weights(weight_func = init_weight_func)
        self.net.save_weights()
        for t in test:
            self.net.reload_weights()
            t()
            print '\nTest for [%s]:\n' % t.__name__
            self.run(epoches = 10)

    def test_CrossEntropy_vs_Quadratic(self):
        def cost_quadratic(): self.net.set(cost_func = Quadratic)
        def cost_cross_entropy(): self.net.set(cost_func = CrossEntropy)
        self.run_test([cost_quadratic, cost_cross_entropy], init_weight_func = WeightRandom(0.5))

    def test_softmax(self):
        def cost_cross_entropy(): self.net.set(cost_func = CrossEntropy)
        def last_layer_softmax(): self.net.set(cost_func = Loglikelihood, last_layer_activation_func = Softmax)
        self.run_test([last_layer_softmax, cost_cross_entropy], init_weight_func = WeightRandom(0.5))

    def test_ReLU(self):
        def default(): self.net.set()
        def act_ReLU(): self.net.set(activation_func = ReLU)
        self.run_test([act_ReLU, default], init_weight_func = WeightRandom(1))


if __name__ == '__main__':
    test = TestNetwork()
    #test.test_CrossEntropy_vs_Quadratic()
    #test.test_softmax()
    test.test_ReLU()

