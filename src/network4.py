#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import time, sys, json
import pprint

# activation function: sigmoid, softmax(output layer), tanh, rectified linear(ReLU)
# cost function: quadratic, cross-entropy
# regularization: none, L2, L1, dropout, artificially expanding training data
# weight initialization: all zero, random, random with good variance
# stochastic gradient descent: momentum-based

# hyper-parameters: learning rate, epoch (early stopping), regularization lambda, mini-batch size

class Debug(object):
    ENABLE = False
    OUTPUT_FILE = ''

    @staticmethod
    def output_file(filename):
        Debug.OUTPUT_FILE = filename
        if filename: open(filename, 'w').close()

    @staticmethod
    def print_(*str):
        if Debug.ENABLE:
            for s in str:
                pp = pprint.PrettyPrinter(width = 200, depth = 10, stream = open(Debug.OUTPUT_FILE, 'a') if Debug.OUTPUT_FILE else None)
                pp.pprint(s)
                sys.stdout.flush()

class Sigmoid(object):
    @staticmethod
    def f(z):
        return 1.0/(1.0 + np.exp(-z))

    @staticmethod
    def derivative(z):
        a = Sigmoid.f(z)
        return a * (1 - a)

class Quadratic(object):
    @staticmethod
    def derivative(a, y):
        return (a - y)

class WeightRandom(object):
    @staticmethod
    def init(sizes):
        num_layers = len(sizes)
        # weights: num_layers-1 elements, each element is m (the latter layer) * n (the former layer)
        weights = [np.random.randn(sizes[layer], sizes[layer-1]) for layer in range(1, num_layers)]
        # biases: num_layer-1 elements, each element is n (the current layer)
        biases = [np.random.randn(sizes[layer], 1) for layer in range(1, num_layers)]
        return weights, biases

class WeightConstant(object):
    @staticmethod
    def init(sizes):
        num_layers = len(sizes)
        weights = [0.5*np.ones((sizes[layer], sizes[layer-1])) for layer in range(1, num_layers)]
        biases = [0.5*np.ones((sizes[layer], 1)) for layer in range(1, num_layers)]
        return weights, biases

class Network(object):
    def __init__(self, sizes):
        self.init(sizes)

    def init(self, sizes):
        # sizes, number of neurons of [input_layer, hidden layer, ..., output_layer]
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.set()

    def set(self, activation_func = Sigmoid, cost_func = Quadratic, regularization = None, weight_func = WeightRandom):
        self.act_func = activation_func
        self.cost_func = cost_func
        self.regularization = regularization
        self.weight_func = weight_func
        self.weights, self.biases = self.weight_func.init(self.sizes)

    def feedforward(self, a):
        (a, _, _) = self.feedforward_layers(a)
        return a

    def feedforward_layers(self, a):
        a_layers = [a]
        z_layers = [None]
        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(weight, a) + bias
            a = self.act_func.f(z)
            a_layers.append(a)
            z_layers.append(z)
        return (a, a_layers, z_layers)

    def back_propogation(self, x, y):
        a, a_layers, z_layers = self.feedforward_layers(x)
        Debug.print_('feedforward: x:', x, 'a:', a)
        delta_w, delta_b = [], []
        for layer in range(self.num_layers-1, 0, -1):
            if layer == self.num_layers-1:
                delta = self.cost_func.derivative(a, y)
            else:
                delta = np.dot(self.weights[layer].transpose(), delta)
            delta *= self.act_func.derivative(z_layers[layer])  # delta for layer
            delta_w.append(np.dot(delta, a_layers[layer-1].transpose()))
            delta_b.append(delta)
        delta_w.reverse()
        delta_b.reverse()
        return delta_w, delta_b

    def mini_batch_update(self, training_data, learning_rate):
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]
        Debug.print_('delta_w:', delta_w, 'delta_b:', delta_b)
        for x, y in training_data:
            Debug.print_('x:', x, 'y:', y)
            delta_w_i, delta_b_i = self.back_propogation(x, y)
            Debug.print_('delta_w_i:', delta_w_i, 'delta_b_i:', delta_b_i)
            delta_w = [w + w_i for w, w_i in zip(delta_w, delta_w_i)]
            delta_b = [b + b_i for b, b_i in zip(delta_b, delta_b_i)]
        Debug.print_('delta_w:', delta_w, 'delta_b:', delta_b)
        Debug.print_('weights:', self.weights, 'biases:', self.biases)
        self.weights = [w - learning_rate / len(training_data) * dw for w, dw in zip(self.weights, delta_w)]
        self.biases = [d - learning_rate / len(training_data) * dd for d, dd in zip(self.biases, delta_b)]
        Debug.print_('weights:', self.weights, 'biases:', self.biases)
        Debug.ENABLE = False

    # stochastic gradient descent
    def sgd(self, training_data, epoch, mini_batch_size, learning_rate, test_data = []):
        # training_data, [(x0, y0), (x1, y1), ...]
        time_start = time.time()
        for t in range(epoch):
            np.random.shuffle(training_data)
            start = 0
            while start < len(training_data):
                mini_batch_data = training_data[start:min(start+mini_batch_size, len(training_data))]
                start += mini_batch_size
                self.mini_batch_update(mini_batch_data, learning_rate)
            if test_data:
                print 'epoch %d: accurate rate %.2f%%, elapsed: %.1fs' % (t, 100*self.test(test_data), time.time() - time_start)

    def test(self, test_data):
        num_pass = sum([y == np.argmax(self.feedforward(x)) for x, y in test_data])
        return num_pass * 1.0 / len(test_data)

    def save(self, filename):
        data = {'sizes': self.sizes,
                'weights': [w.tolist() for w in self.weights],
                'biases': [b.tolist() for b in self.biases],
                'act_func': str(self.act_func.__name__),
                'cost_func': str(self.cost_func.__name__)}
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load(self, filename):
        data = json.load(open(filename, 'r'))
        self.init(data['sizes'])
        self.weights = [np.array(w) for w in data['weights']]
        self.biases = [np.array(b) for b in data['biases']]


if __name__ == '__main__':
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, 30, 10])
    #net.sgd(training_data, 2, 10, 3.0, test_data = test_data)
    #net.save('network.txt')

    #net.set(weight_func = WeightConstant)
    Debug.ENABLE = True
    Debug.output_file('output4.txt')
    net.load('network.txt')
    net.sgd(training_data, 1, 10, 3.0, test_data=test_data)
    #net.sgd(training_data, 30, 10, 3.0, test_data=test_data)



