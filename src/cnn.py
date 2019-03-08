#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import time, sys, json
import pprint


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
        weights = [np.random.randn(sizes[layer], sizes[layer-1]) / np.sqrt(sizes[layer-1]) for layer in range(1, num_layers)]
        #weights = [np.random.randn(sizes[layer], sizes[layer-1]) / sizes[layer-1] for layer in range(1, num_layers)]
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

########### train method ####################
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

########### early stopping ####################
class EarlyStop(object):
    def __init__(self, check_epoches = 10):
        # for some epoches that the test accuracy do not get a new larger value
        self.check_epoches = check_epoches
        self.init()

    def init(self):
        self.accuracy_history = []

    def stop(self, test_accuracy = None):
        self.accuracy_history.append(test_accuracy)
        if len(self.accuracy_history) >= self.check_epoches:
            last_accuracy = self.accuracy_history[-self.check_epoches:]
            return np.argmax(last_accuracy) == 0
        return False

class EpochStop(object):
    def __init__(self, stop_epoches = 30):
        self.stop_epoches = stop_epoches

    def init(self):
        self.run_epoches = 0

    def stop(self, test_accuracy = None):
        self.run_epoches += 1
        if (self.run_epoches >= self.stop_epoches):
            self.run_epoches = 0
            return True
        return False

########### convolutional neural network ####################
class ConvLayer(object):
    def __init__(self, sizes, activation_func = ReLU):
        # (i, j, q, p), i*j filter matrix, q is the deep length of input data, p is the deep length of output data
        self.filter_shape, self.input_deep, self.output_deep = (sizes[0], sizes[1]), sizes[2], sizes[3]
        self.act_func = activation_func
        # 4D, p * q * (i * j)
        self.weights = np.random.randn(self.output_deep, self.input_deep, *self.filter_shape)
        # 4D, p * 1 * 1
        self.biases = np.zeros((self.output_deep, 1, 1))
        self.a_for_back_propogation = None
        self.delta_w = self.delta_b = None

    def conv2d(self, a, f):
        # refer: https://stackoverflow.com/questions/43086557/convolve2d-just-by-using-numpy
        # a: 3D matrix q *(m * n), f: filter weights, 4D matrix (include deep) (p * q * (i * j))
        # output: 3D matrix (include deep) (p * (m-i+1) * (n-j+1))
        assert a.ndim == 3, 'invalid dimention of a'
        assert f.ndim == 4, 'invlaid dimention of f'
        view_shape = tuple(np.subtract(a.shape[1:], f.shape[2:]) + 1) + (a.shape[0],) + f.shape[2:]
        view_strides = a.strides[1:] + a.strides
        # ((m-i+1) * (n-j+1) * q * i * j)
        view_matrix = np.lib.stride_tricks.as_strided(a, view_shape, view_strides, writeable = False)
        return np.einsum('pqij,klqij->pkl', f, view_matrix)  # p is deep length of output

    def back_conv2d(self, a, f):
        # a: 3D matrix q *(m * n),
        # f: 3D matrix (include deep) (p * (m-i+1) * (n-j+1))
        # output: filter weights, 4D matrix (include deep) (p * q * (i * j))
        assert a.ndim == 3, 'invalid dimention of a'
        assert f.ndim == 3, 'invlaid dimention of f'
        view_shape = tuple(np.subtract(a.shape[1:], f.shape[1:]) + 1) + (a.shape[0],) + f.shape[1:]
        view_strides = a.strides[1:] + a.strides
        # (i * j * q * (m-i+1) * (n-j+1))
        view_matrix = np.lib.stride_tricks.as_strided(a, view_shape, view_strides, writeable = False)
        return np.einsum('pkl,ijqkl->pqij', f, view_matrix)

    def feedforward(self, a, in_back_propogation = False):
        assert a.shape[0] == self.input_deep, 'invalid input deep'
        z = self.conv2d(a, self.weights) + self.biases
        a_output = self.act_func.f(z)
        if in_back_propogation: self.a_for_back_propogation = [a, a_output]
        return a_output

    def back_propogation(self, delta):
        assert delta.ndim == 3, 'invalid dimension for delta'
        assert delta.shape[0] == self.output_deep, 'invalid output deep'
        assert delta.shape == self.a_for_back_propogation[1].shape, 'invalid shape'
        # delta: 3D matrix (include deep) (p * (m-i+1) * (n-j+1))
        delta[self.a_for_back_propogation[1] <= 0] = 0  # back for ReLU
        delta_b = delta.sum(axis = (1,2))
        delta_w = self.back_conv2d(self.a_for_back_propogation[0], delta)
        #Debug.ENABLE = True
        #Debug.print_('a_for_back_propogation:', self.a_for_back_propogation, 'delta:', delta, 'delta_w:', delta_w)
        self.delta_b = delta_b if self.delta_b is None else self.delta_b + delta_b
        self.delta_w = delta_w if self.delta_w is None else self.delta_w + delta_w
        pad_len = self.filter_shape[0] - 1
        # delta_pad: (p * (m+i-1) * (n+j-1))
        delta_pad = np.pad(delta, ((0, 0), (pad_len, pad_len), (pad_len, pad_len)), 'constant', constant_values = 0)
        # weights: 4D matrix (include deep) (p * q * (i * j)), q is the deep length of input data, p is the deep length of output data
        back_weights = np.rot90(self.weights, 2, axes = (2,3))
        # turn to: (q * p * (i * j))
        back_weights = np.transpose(back_weights, axes = [1, 0, 2, 3])
        # input: delta_pad: (p * (m+i-1) * (n+j-1)), f: filter weights, 4D matrix (include deep) (q * p * (i * j))
        # output: like origin a, 3D matrix q *(m * n)
        z = self.conv2d(delta_pad, back_weights)
        return z
        
    def update_weights(self, mini_batch_data_size, training_size):
        learning_rate = 0.3
        self.weights = self.weights - learning_rate / mini_batch_data_size * self.delta_w
        self.biases = self.biases - learning_rate / mini_batch_data_size * self.delta_b
        self.delta_w = self.delta_b = None

class PoolingLayer(object):
    def __init__(self, strides = 2):
        self.strides = strides

    def feedforward(self, a, in_back_propogation = False):
        # MAX pooling
        # a: 3D matrix q *(m * n)
        assert a.ndim == 3, 'invalid dimention of a'
        view_shape = (a.shape[0], a.shape[1]/self.strides, a.shape[2]/self.strides, self.strides, self.strides)
        view_strides = (a.strides[0], a.strides[1]*self.strides, a.strides[2]*self.strides, a.strides[1], a.strides[2])
        view_matrix = np.lib.stride_tricks.as_strided(a, view_shape, view_strides, writeable = False)
        # 5D matrix, q * m/stride * n/stride * stride * stride
        a =  view_matrix.max(axis = (3,4))
        # save argmax index for back propogation
        c = view_matrix.reshape(view_matrix.shape[0], view_matrix.shape[1], view_matrix.shape[2], -1)
        self.index_for_back_propogation = np.argmax(c, axis = -1)
        return a

    def back_propogation(self, delta):
        assert delta.ndim == 3, 'invalid dimention of delta'
        delta_shape = (delta.shape[0], delta.shape[1], delta.shape[2], self.strides*self.strides)
        z = np.zeros(delta_shape).reshape(-1, delta_shape[-1])
        z[range(z.shape[0]), np.ravel(self.index_for_back_propogation)] = np.ravel(delta)
        z = z.reshape(delta.shape[0], delta.shape[1]*self.strides, delta.shape[2]*self.strides)
        return z
        
    def update_weights(self, mini_batch_data_size, training_size):
        pass

class FcLayer(object):
    def __init__(self, sizes):
        self.init(sizes)
        self.init_weights()
        self.delta_w = self.delta_b = None

    def init(self, sizes):
        # sizes, number of neurons of [input_layer, hidden layer, ..., output_layer]
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.set_neuron()
        self.set_regularization()
        self.set_train_func()
        self.set_dropout()
        self.set_stop()

    def init_weights(self, weight_func = WeightOpt()):
        self.weights, self.biases = weight_func.init(self.sizes)
        self.saved_weights = (self.weights, self.biases)

    def reload_weights(self):
        assert hasattr(self, 'saved_weights'), 'no saved weights'
        self.weights, self.biases = self.saved_weights

    def ready_for_train(self):
        if hasattr(self.train_func, 'init'): self.train_func.init(self.sizes)
        if hasattr(self.stop, 'init'): self.stop.init()

    def set_neuron(self, activation_func = Sigmoid, last_layer_activation_func = None, cost_func = CrossEntropy):
        self.act_func = activation_func
        self.last_layer_act_func = last_layer_activation_func or self.act_func
        self.cost_func = cost_func

    def set_regularization(self, regularization = RegularNone()):
        self.regularization = regularization

    def set_train_func(self, train_func = Sgd(0.1)):
        self.train_func = train_func

    def set_dropout(self, dropout = None):
        self.dropout = dropout

    def set_stop(self, stop = EarlyStop(30)):
        self.stop = stop

    def feedforward(self, a, in_back_propogation = False):
        a = a.reshape(-1, 1)
        if in_back_propogation: self.a_layers_for_back_propogation = [a]
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            is_last_layer = (i == (self.num_layers - 2))
            if self.dropout and not in_back_propogation: weight = self.dropout.adjust_weight(weight)
            z = np.dot(weight, a) + bias
            if is_last_layer:
                a = self.last_layer_act_func.f(z)
            else:   # hidden layer
                a = self.act_func.f(z)
                if self.dropout and in_back_propogation: self.dropout.process(z, a)
            if in_back_propogation: self.a_layers_for_back_propogation.append(a)
        return a

    def back_propogation(self, y):
        delta_w, delta_b = [], []
        for layer in range(self.num_layers)[::-1]:
            if layer == self.num_layers-1:  # the last layer
                delta = self.cost_func.delta(self.a_layers_for_back_propogation[layer], y, self.last_layer_act_func)
            else:
                delta = np.dot(self.weights[layer].T, delta)
                #delta *= self.act_func.derivative(z_layers[layer])  # delta for layer
                delta *= self.act_func.derivative_a(self.a_layers_for_back_propogation[layer])  # delta for layer
            if layer > 0:
                delta_w.append(np.dot(delta, self.a_layers_for_back_propogation[layer-1].T))
                delta_b.append(np.dot(delta, np.ones((delta.shape[1],1))))
        delta_w.reverse()
        delta_b.reverse()
        self.delta_b = delta_b if self.delta_b is None else [b+db for b, db in zip(self.delta_b, delta_b)]
        self.delta_w = delta_w if self.delta_w is None else [w+dw for w, dw in zip(self.delta_w, delta_w)]
        size = int(np.sqrt(delta.size/1))
        assert size == 13, 'invalid size'
        delta = delta.reshape(1, size, size)
        return delta

    def update_weights(self, mini_batch_data_size, training_size):
        learning_rate = 0.3
        self.weights = [self.regularization.update_weights(w, learning_rate, training_size) - learning_rate / mini_batch_data_size * dw for w, dw in zip(self.weights, self.delta_w)]
        self.biases = [b - learning_rate / mini_batch_data_size * db for b, db in zip(self.biases, self.delta_b)]
        self.delta_w = self.delta_b = None

class ConvNetwork(object):
    def __init__(self, input_sizes, conv_sizes, fc_sizes):
        # input_sizes: [input layer, m*n*q (input deep)], conv_sizes: [i,j,p (output deep)], fc_sizes: [full connected hidden layer, output layer]
        self.input_sizes, self.input_deep = input_sizes[:2], input_sizes[2]
        self.conv_layer = ConvLayer([conv_sizes[0], conv_sizes[1], self.input_deep, conv_sizes[2]])
        pooling_strides = 2
        self.pooling_layer = PoolingLayer(pooling_strides)
        fc_input_size = (input_sizes[0]-conv_sizes[0]+1) * (input_sizes[1]-conv_sizes[1]+1) * conv_sizes[2] / pooling_strides / pooling_strides
        self.fc_layer = FcLayer([fc_input_size] + fc_sizes)
        self.layers = [self.conv_layer, self.pooling_layer, self.fc_layer]

    @classmethod
    def unpack_data(cls, data):
        nx, ny, data_size = data[0][0].size, data[0][1].size, len(data)
        data_x = np.array([x for x, y in data]).reshape((data_size, nx)).T
        data_y = np.array([y for x, y in data]).reshape((data_size, ny)).T
        return (data_x, data_y)

    def feedforward(self, a, in_back_propogation = False):
        for layer in self.layers:
            a = layer.feedforward(a, in_back_propogation)
        return a
        
    def back_propogation(self, delta):
        for layer in self.layers[::-1]:
            delta = layer.back_propogation(delta)
        return delta
        
    def update_weights(self, mini_batch_data_size, training_size):
        for layer in self.layers:
            layer.update_weights(mini_batch_data_size, training_size)
        
    def mini_batch_update(self, mini_batch_data, training_size):
        # training for the whole mini_batch data once
        #mini_batch_data = self.unpack_data(mini_batch_data)
        for x, y in mini_batch_data:
            self.feedforward(x, in_back_propogation = True)
            self.back_propogation(y)
            self.update_weights(len(mini_batch_data), training_size)
        #self.weights, self.biases = self.train_func.update_weights(self.weights, self.biases, delta_w, delta_b, len(mini_batch_data), training_size, self.regularization)
        #Debug.print_('weights:', self.weights, 'biases:', self.biases)

    # stochastic gradient descent
    def train(self, training_data, mini_batch_size, test_data = []):
        def print_training_info(test_data_accuracy = None):
            if not hasattr(print_training_info, 'training_epoch'): print_training_info.training_epoch = -1
            print_training_info.training_epoch += 1
            training_data_accuracy = 100*self.accuracy(training_data, convert = True)
            training_data_cost = self.cost(training_data)
            if test_data_accuracy:
                print 'epoch %d: cost %.3f training accuracy %.2f%%, test accuracy %.2f%%, elapsed: %.1fs' \
                    % (print_training_info.training_epoch, training_data_cost, training_data_accuracy, 100*test_data_accuracy, time.time() - time_start)
            else:
                print 'epoch %d: cost %.3f training accuracy %.2f%%, elapsed: %.1fs' \
                    % (print_training_info.training_epoch, training_data_cost, training_data_accuracy, time.time() - time_start)
        #self.ready_for_train()
        # training_data, [(x0, y0), (x1, y1), ...]
        training_size = len(training_data)
        time_start = time.time()
        test_data_accuracy = self.accuracy(test_data) if test_data else None
        print_training_info(test_data_accuracy)
        while not self.fc_layer.stop.stop(test_data_accuracy):
            np.random.shuffle(training_data)
            start = 0
            while start < training_size:
                mini_batch_data = training_data[start:min(start+mini_batch_size, training_size)]
                start += mini_batch_size
                self.mini_batch_update(mini_batch_data, training_size)
            test_data_accuracy = self.accuracy(test_data) if test_data else None
            print_training_info(test_data_accuracy)

    def cost(self, training_data):
        return sum([self.fc_layer.cost_func.cost(self.feedforward(x), y) for x, y in training_data])

    def accuracy(self, test_data, convert = False):
        if convert:
            num_pass = sum([(np.argmax(self.feedforward(x), axis = 0) == np.argmax(y, axis = 0)) for x, y in test_data])
        else:
            num_pass = sum([(np.argmax(self.feedforward(x), axis = 0) == y) for x, y in test_data])
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
    # 50000, 10000, 10000
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = training_data[:1000]
    test_data = test_data[:1000]
    training_data = [(x.reshape(1, 28, 28), y) for x, y in training_data]
    test_data = [(x.reshape(1, 28, 28), y) for x, y in test_data]
    # input_sizes: [input layer, m*n*q (input deep)], conv_sizes: [i,j,p (output deep)], fc_sizes: [full connected hidden layer, output layer]        
    net = ConvNetwork([28, 28, 1], [3, 3, 1], [30, 10])
    net.train(training_data, 30, test_data = test_data)



