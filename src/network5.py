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

    @classmethod
    def output_file(cls, filename):
        cls.OUTPUT_FILE = filename
        if filename: open(filename, 'w').close()

    @classmethod
    def print_(cls, *str):
        if cls.ENABLE:
            for s in str:
                pp = pprint.PrettyPrinter(width = 200, depth = 10, stream = open(cls.OUTPUT_FILE, 'a') if cls.OUTPUT_FILE else None)
                pp.pprint(s)
                sys.stdout.flush()

########### activation function ####################
class Sigmoid(object):
    @staticmethod
    def f(z):
        return 1.0/(1.0 + np.exp(-z))

    @staticmethod
    def derivative(z):
        a = f.__func__(z)
        return a * (1 - a)

    @staticmethod
    def derivative_a(a):
        return a * (1 - a)

class Tanh(object):
    @staticmethod
    def f(z):
        t1 = np.exp(z)
        t2 = np.exp(-z)
        return (t1 - t2) / (t1 + t2)

    @staticmethod
    def derivative_a(a):
        return 1 - a * a

# seems when using ReLU, small standardized initial weights and zero initial biases should be used
# also smaller learning rate
# the performance greatly depends on the initial weights, seems some local minimum exist???
class ReLU(object):
    @staticmethod
    def f(z):
        temp = z.copy()
        temp[temp < 0] = 0
        return temp

    @staticmethod
    def derivative_a(a):
        temp = a.copy()
        temp[temp <= 0] = 0
        temp[temp > 0] = 1
        return temp

class Softmax(object):
    @staticmethod
    def f(z):
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis = 0)

########### cost function ####################
class Loglikelihood(object):
    @staticmethod
    def cost(a, y):
        return -(y * np.log(a)).sum(axis = 0).mean()

    @staticmethod
    def delta(a, y, act_func = Softmax):
        assert act_func == Softmax, 'only Softmax is supported with Loglikelihood'
        return a - y

class Quadratic(object):
    @staticmethod
    def cost(a, y):
        return np.square(a - y).mean()

    @staticmethod
    def derivative(a, y):
        return (a - y)

    @staticmethod
    def delta(a, y, act_func = Sigmoid):
        return (a - y) * act_func.derivative_a(a)

class CrossEntropy(object):
    @staticmethod
    def cost(a, y):
        return -(y * np.log(a) + (1 - y) * np.log(1 - a)).mean()

    @staticmethod
    def derivative(a, y):
        return (a - y) / ( a * (1 - a))

    @staticmethod
    def delta(a, y, act_func = Sigmoid):
        assert act_func == Sigmoid, 'only Sigmoid is supported with CrossEntropy'
        return (a - y)

########### regularization ####################
class RegularNone(object):
    def cost(self, weights, data_size):
        return 0

    def update_weights(self, weights, learning_rate, total_training_size):
        return weights

class RegularL2(object):
    def __init__(self, lmda = 0.1):
        self.lmda = lmda

    def cost(self, weights, data_size):
        return self.lmda / 2 / data_size * sum([np.square(w).sum() for w in weights])

    def update_weights(self, weights, learning_rate, total_training_size):
        return (1 - self.lmda * learning_rate / total_training_size) * weights

class RegularL1(object):
    def __init__(self, lmda = 0.1):
        self.lmda = lmda

    def cost(self, weights, data_size):
        return self.lmda / data_size * sum([np.abs(w).sum() for w in weights])

    def update_weights(self, weights, learning_rate, total_training_size):
        return weights - self.lmda * learning_rate / total_training_size * np.sign(weights)

class Dropout(object):
    def __init__(self, drop_probability = 0.5):
        self.drop_probability = drop_probability

    def process(self, z, a):
        assert z.size == a.size, 'invalid size of a or z'
        index = range(z.shape[0])
        np.random.shuffle(index)
        drop_index = index[:int(len(index) * self.drop_probability)]
        z[drop_index] = 0   # drop the neuron is equivalent to set z and a to 0
        a[drop_index] = 0
        
    def adjust_weight(self, weight):
        return weight * (1-self.drop_probability)

########### weight initialize function ####################
class WeightOpt(object):
    def init(self, sizes):
        num_layers = len(sizes)
        # weights: num_layers-1 elements, each element is m (the latter layer) * n (the former layer) matrix
        # biases: num_layer-1 elements, each element is n (the current layer) * 1 matrix
        #weights = [np.random.randn(sizes[layer], sizes[layer-1]) / np.sqrt(sizes[layer-1]) for layer in range(1, num_layers)]
        weights = [np.random.randn(sizes[layer], sizes[layer-1]) / sizes[layer-1] for layer in range(1, num_layers)]
        biases = [np.zeros((sizes[layer], 1)) for layer in range(1, num_layers)]
        return weights, biases

class WeightRandom(object):
    def __init__(self, mean = 0, bias_zero = False, large = True):
        self.mean = mean
        self.bias_zero = bias_zero
        self.large = large

    def init(self, sizes):
        num_layers = len(sizes)
        if self.large:
            weights = [self.mean + np.random.randn(sizes[layer], sizes[layer-1]) for layer in range(1, num_layers)]
        else:
            weights = [self.mean + np.random.randn(sizes[layer], sizes[layer-1]) / np.sqrt(sizes[layer-1]) for layer in range(1, num_layers)]
        if self.bias_zero:
            biases = [np.zeros((sizes[layer], 1)) for layer in range(1, num_layers)]
        else:
            if self.large:
                biases = [self.mean + np.random.randn(sizes[layer], 1) for layer in range(1, num_layers)]
            else:
                biases = [self.mean + np.random.randn(sizes[layer], 1) / np.sqrt(sizes[layer]) for layer in range(1, num_layers)]
        return weights, biases

class WeightConstant(object):
    def __init__(self, w = 0, b = 0):
        self.constant_w = w
        self.constant_b = b

    def init(sizes):
        num_layers = len(sizes)
        weights = [self.constant_w * np.ones((sizes[layer], sizes[layer-1])) for layer in range(1, num_layers)]
        biases = [self.constant_b * np.ones((sizes[layer], 1)) for layer in range(1, num_layers)]
        return weights, biases

class MomentumSgd(object):
    def __init__(self, learning_rate = 0.1, coeffient = 0.5):
        self.learning_rate = learning_rate
        self.coeffient = coeffient

    def init(self, sizes):
        num_layers = len(sizes)
        self.weights_velocity = [np.zeros((sizes[layer], sizes[layer-1])) for layer in range(1, num_layers)]
        self.biases_velocity = [np.zeros((sizes[layer], 1)) for layer in range(1, num_layers)]

    def update_weights(self, weights, biases, delta_w, delta_b, data_size, total_training_size, regularization):
        self.weights_velocity = [self.coeffient * wv - self.learning_rate / data_size * dwv for wv, dwv in zip(self.weights_velocity, delta_w)]
        self.biases_velocity = [self.coeffient * bv - self.learning_rate / data_size * dbv for bv, dbv in zip(self.biases_velocity, delta_b)]
        weights = [regularization.update_weights(w, self.learning_rate, total_training_size) + wv for w, wv in zip(weights, self.weights_velocity)]
        biases = [b + bv for b, bv in zip(biases, self.biases_velocity)]
        return weights, biases

class Sgd(object):
    def __init__(self, learning_rate = 0.1):
        self.learning_rate = learning_rate

    def update_weights(self, weights, biases, delta_w, delta_b, data_size, total_training_size, regularization):
        weights = [regularization.update_weights(w, self.learning_rate, total_training_size) - self.learning_rate / data_size * dw for w, dw in zip(weights, delta_w)]
        biases = [b - self.learning_rate / data_size * db for b, db in zip(biases, delta_b)]
        return weights, biases

########### neural network ####################
class Network(object):
    def __init__(self, sizes):
        self.init(sizes)
        self.init_weights()

    def init(self, sizes):
        # sizes, number of neurons of [input_layer, hidden layer, ..., output_layer]
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.set_neuron()
        self.set_regularization()
        self.set_train_func()
        self.set_dropout()

    def init_weights(self, weight_func = WeightOpt()):
        self.weights, self.biases = weight_func.init(self.sizes)
        self.saved_weights = (self.weights, self.biases)
        if hasattr(self.train_func, 'init'): self.train_func.init(self.sizes)

    def reload_weights(self):
        assert hasattr(self, 'saved_weights'), 'no saved weights'
        self.weights, self.biases = self.saved_weights
        if hasattr(self.train_func, 'init'): self.train_func.init(self.sizes)

    def set_neuron(self, activation_func = Sigmoid, last_layer_activation_func = None, cost_func = Quadratic):
        self.act_func = activation_func
        self.last_layer_act_func = last_layer_activation_func or self.act_func
        self.cost_func = cost_func

    def set_regularization(self, regularization = RegularNone()):
        self.regularization = regularization

    def set_train_func(self, train_func = Sgd(0.1)):
        self.train_func = train_func

    def set_dropout(self, dropout = None):
        self.dropout = dropout

    def feedforward(self, a):
        (a, _, _) = self.feedforward_layers(a, enable_dropout = False)
        return a

    def feedforward_layers(self, a, enable_dropout = False):
        a_layers = [a]
        z_layers = [None]
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            is_last_layer = (i == (self.num_layers - 2))
            if self.dropout and not enable_dropout: weight = self.dropout.adjust_weight(weight)
            z = np.dot(weight, a) + bias
            if is_last_layer:
                a = self.last_layer_act_func.f(z) 
            else:   # hidden layer
                a = self.act_func.f(z)
                if self.dropout and enable_dropout: self.dropout.process(z, a)
            a_layers.append(a)
            z_layers.append(z)
        return (a, a_layers, z_layers)

    def back_propogation(self, x, y):
        a, a_layers, z_layers = self.feedforward_layers(x, enable_dropout = True)
        delta_w, delta_b = [], []
        for layer in range(self.num_layers-1, 0, -1):
            if layer == self.num_layers-1:  # the last layer
                delta = self.cost_func.delta(a, y, self.last_layer_act_func)
            else:
                delta = np.dot(self.weights[layer].transpose(), delta)
                #delta *= self.act_func.derivative(z_layers[layer])  # delta for layer
                delta *= self.act_func.derivative_a(a_layers[layer])  # delta for layer
            delta_w.append(np.dot(delta, a_layers[layer-1].transpose()))
            delta_b.append(np.dot(delta, np.ones((delta.shape[1],1))))
        delta_w.reverse()
        delta_b.reverse()
        return delta_w, delta_b

    @classmethod
    def unpack_data(cls, data):
        nx, ny, data_size = data[0][0].size, data[0][1].size, len(data)
        data_x = np.array([x for x, y in data]).reshape((data_size, nx)).transpose()
        data_y = np.array([y for x, y in data]).reshape((data_size, ny)).transpose()
        return (data_x, data_y)

    def mini_batch_update(self, mini_batch_data, training_size):
        # training for the whole mini_batch data once
        mini_batch_data = self.unpack_data(mini_batch_data)
        delta_w, delta_b = self.back_propogation(*mini_batch_data)
        self.weights, self.biases = self.train_func.update_weights(self.weights, self.biases, delta_w, delta_b, len(mini_batch_data), training_size, self.regularization)
        #Debug.print_('weights:', self.weights, 'biases:', self.biases)

    # stochastic gradient descent
    def train(self, training_data, epoch, mini_batch_size, test_data = []):
        def output_info(epoch_i):
            test_data_accuracy = 100*self.accuracy(test_data)
            training_data_accuracy = 100*self.accuracy(training_data, convert = True)
            training_data_cost = self.cost(training_data)
            print 'epoch %d: cost %.3f accurate rate %.2f%%, %.2f%%, elapsed: %.1fs' % (epoch_i, training_data_cost, training_data_accuracy, test_data_accuracy, time.time() - time_start)
        # training_data, [(x0, y0), (x1, y1), ...]
        training_size = len(training_data)
        time_start = time.time()
        if test_data: output_info(0)
        for t in range(epoch):
            np.random.shuffle(training_data)
            start = 0
            while start < training_size:
                mini_batch_data = training_data[start:min(start+mini_batch_size, training_size)]
                start += mini_batch_size
                self.mini_batch_update(mini_batch_data, training_size)
            if test_data: output_info(t+1)

    def cost(self, training_data):
        data = self.unpack_data(training_data)
        return self.cost_func.cost(self.feedforward(data[0]), data[1]) + self.regularization.cost(self.weights, len(training_data))

    def accuracy(self, test_data, convert = False):
        if convert:
            test_data = self.unpack_data(test_data)
            test_data = (test_data[0], np.argmax(test_data[1], axis = 0))
        num_pass = (np.argmax(self.feedforward(test_data[0]), axis = 0) == test_data[1]).sum()
        return num_pass * 1.0 / test_data[1].size

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
    #net.save('network.txt')
    net.set(cost_func = CrossEntropy)
    #Debug.ENABLE = True
    #Debug.output_file('output5.txt')
    #net.load('network.txt')
    net.train(training_data, 30, 10, 3.0, test_data = test_data)



