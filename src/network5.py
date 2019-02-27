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

    @staticmethod
    def derivative_a(a):
        return a * (1 - a)

class ReLU(object):
    @staticmethod
    def f(z):
        z[z < 0] = 0
        return z

    @staticmethod
    def derivative_a(a):
        a[a <= 0] = 0
        a[a > 0] = 1
        return a

class Softmax(object):
    @staticmethod
    def f(z):
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis = 0)

class Loglikelihood(object):
    @staticmethod
    def delta(a, y):  # with Softmax
        return a - y

class Quadratic(object):
    @staticmethod
    def derivative(a, y):
        return (a - y)

    @staticmethod
    def delta(a, y):  # with Sigmoid
        return (a - y) * Sigmoid.derivative_a(a)

class CrossEntropy(object):
    @staticmethod
    def derivative(a, y):
        return (a - y) / ( a * (1 - a))

    @staticmethod
    def delta(a, y):  # with Sigmoid
        return (a - y)

class WeightRandom(object):
    def __init__(self, mean = 0):
        self.mean = mean

    def init(self, sizes):
        num_layers = len(sizes)
        # weights: num_layers-1 elements, each element is m (the latter layer) * n (the former layer)
        weights = [self.mean + np.random.randn(sizes[layer], sizes[layer-1]) for layer in range(1, num_layers)]
        # biases: num_layer-1 elements, each element is n (the current layer)
        biases = [self.mean + np.random.randn(sizes[layer], 1) for layer in range(1, num_layers)]
        return weights, biases

class WeightConstant(object):
    @staticmethod
    def init(sizes):
        num_layers = len(sizes)
        weights = [np.ones((sizes[layer], sizes[layer-1])) for layer in range(1, num_layers)]
        biases = [np.ones((sizes[layer], 1)) for layer in range(1, num_layers)]
        return weights, biases

class Network(object):
    def __init__(self, sizes):
        self.init(sizes)
        self.init_weights()

    def init(self, sizes):
        # sizes, number of neurons of [input_layer, hidden layer, ..., output_layer]
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.set()

    def init_weights(self, weight_func = WeightRandom(), weights = None, biases = None):
        if weights and biases:
            self.weights, self.biases = weights, biases
        else:
            self.weights, self.biases = weight_func.init(self.sizes)

    def save_weights(self):
        self.saved_weights = (self.weights, self.biases)

    def reload_weights(self):
        if hasattr(self, 'saved_weights'):
            self.weights, self.biases = self.saved_weights
        else:
            print 'no saved weights'

    def set(self, activation_func = Sigmoid, cost_func = Quadratic, last_layer_activation_func = None, regularization = None):
        self.act_func = activation_func
        self.cost_func = cost_func
        self.last_layer_act_func = last_layer_activation_func or self.act_func
        self.regularization = regularization

    def feedforward(self, a):
        (a, _, _) = self.feedforward_layers(a)
        return a

    def feedforward_layers(self, a):
        a_layers = [a]
        z_layers = [None]
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(weight, a) + bias
            a = self.last_layer_act_func.f(z) if i == len(self.biases) else self.act_func.f(z)
            a_layers.append(a)
            z_layers.append(z)
        return (a, a_layers, z_layers)

    def back_propogation(self, x, y):
        a, a_layers, z_layers = self.feedforward_layers(x)
        delta_w, delta_b = [], []
        for layer in range(self.num_layers-1, 0, -1):
            if layer == self.num_layers-1:  # the last layer
                #delta = self.cost_func.derivative(a, y)
                delta = self.cost_func.delta(a, y)
            else:
                delta = np.dot(self.weights[layer].transpose(), delta)
                #delta *= self.act_func.derivative(z_layers[layer])  # delta for layer
                delta *= self.act_func.derivative_a(a_layers[layer])  # delta for layer
            delta_w.append(np.dot(delta, a_layers[layer-1].transpose()))
            delta_b.append(np.dot(delta, np.ones((delta.shape[1],1))))
        delta_w.reverse()
        delta_b.reverse()
        return delta_w, delta_b

    def mini_batch_update_single(self, training_data, learning_rate):
        # training for each training data once
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in training_data:
            delta_w_i, delta_b_i = self.back_propogation(x, y)
            delta_w = [w + w_i for w, w_i in zip(delta_w, delta_w_i)]
            delta_b = [b + b_i for b, b_i in zip(delta_b, delta_b_i)]
        self.weights = [w - learning_rate / len(training_data) * dw for w, dw in zip(self.weights, delta_w)]
        self.biases = [d - learning_rate / len(training_data) * dd for d, dd in zip(self.biases, delta_b)]
        #Debug.print_('weights:', self.weights, 'biases:', self.biases)

    def mini_batch_update(self, training_data, learning_rate):
        # training for the whole mini_batch data once
        nx, ny, data_size = training_data[0][0].size, training_data[0][1].size, len(training_data)
        mini_batch_x = np.array([x for x, y in training_data]).reshape((data_size, nx)).transpose()
        mini_batch_y = np.array([y for x, y in training_data]).reshape((data_size, ny)).transpose()
        delta_w, delta_b = self.back_propogation(mini_batch_x, mini_batch_y)
        self.weights = [w - learning_rate / data_size * dw for w, dw in zip(self.weights, delta_w)]
        self.biases = [d - learning_rate / data_size * dd for d, dd in zip(self.biases, delta_b)]
        #Debug.print_('weights:', self.weights, 'biases:', self.biases)

    # stochastic gradient descent
    def sgd(self, training_data, epoch, mini_batch_size, learning_rate, test_data = []):
        # training_data, [(x0, y0), (x1, y1), ...]
        time_start = time.time()
        if test_data:
            print 'start: accurate rate %.2f%%, elapsed: %.1fs' % (100*self.test(test_data), time.time() - time_start)
        for t in range(epoch):
            np.random.shuffle(training_data)
            start = 0
            while start < len(training_data):
                mini_batch_data = training_data[start:min(start+mini_batch_size, len(training_data))]
                start += mini_batch_size
                self.mini_batch_update(mini_batch_data, learning_rate)
                #self.mini_batch_update_single(mini_batch_data, learning_rate)
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
    net.set(cost_func = CrossEntropy)
    #Debug.ENABLE = True
    #Debug.output_file('output5.txt')
    #net.load('network.txt')
    #net.sgd(training_data, 1, 10, 3.0, test_data = test_data)
    net.sgd(training_data, 30, 10, 3.0, test_data = test_data)



