#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import pprint
from cnn import *


class TestConvLayer(unittest.TestCase):
    def setUp(self):
        #self.conv_layer = ConvLayer([3,3,2,2])
        self.conv_layer = ConvLayer([3,3,1,1])
        
    def tearDown(self):
        pass
        
    def x_test_conv2d(self):
        # a: 2*(6*6), f: 2*2*(3*3)
        a = np.arange(36*2).reshape(2, 6, 6)
        f = np.array([[0,0,1,-1,0,0,0,0,1],
                      [0,0,1,-1,0,0,0,0,1],
                      [1,0,0,0,0,-1,1,0,0],
                      [1,0,0,0,0,-1,1,0,0]]).reshape(2, 2, 3, 3)
        # z: 2*(4*4)
        z = self.conv_layer.conv2d(a, f)
        z1 = np.array([10,11,12,13,16,17,18,19,22,23,24,25,28,29,30,31])*2+36
        z2 = np.array([4,5,6,7,10,11,12,13,16,17,18,19,22,23,24,25])*2+36
        z_expect = np.concatenate((z1,z2)).reshape(2, 4, 4)
        Debug.print_('test_conv2d:', 'z:', z, 'z_expect:', z_expect)
        self.assertTrue((z == z_expect).all())

    def x_test_back_conv2d(self):
        # a: 2*(6*6), f: 2*(4*4)
        a = np.arange(36*2).reshape(2, 6, 6)
        f = np.array([[0,0,0,1,-1,0,0,0,0,0,0,0,1,0,0,0],
                      [1,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,1]]).reshape(2, 4, 4)
        # z: 2*2*(3*3)
        z = self.conv_layer.back_conv2d(a, f)
        z1 = np.array([15,16,17,21,22,23,27,28,29])
        z2 = np.array([12,13,14,18,19,20,24,25,26])
        z_expect = np.concatenate((z1, z1+36, z2, z2+36)).reshape(2, 2, 3, 3)
        Debug.print_('test_back_conv2d:', 'z:', z, 'z_expect:', z_expect)
        self.assertTrue((z == z_expect).all())

    def x_test_feedforward(self):
        pass

    def test_back_propogation(self):
        def back_with_weights(x, tiny_delta_weights, weight_index):
            self.conv_layer = ConvLayer([3,3,1,1])
            self.conv_layer.weights = np.array([[ 0.08305444, -0.16140855,  0.08469047],
                                                [ 1.21509352,  0.47382514, -0.78398948],
                                                [ 0.93826142,  1.3423708 ,  1.84067832]]).reshape(self.conv_layer.weights.shape)
            np.ravel(self.conv_layer.weights)[weight_index] += tiny_delta_weights
            Debug.print_('test_back_propogation:', 'weight index:', weight_index, 'weights:', self.conv_layer.weights)
            #self.conv_layer.biases = origin_weights + tiny_delta_weights
            z = self.conv_layer.feedforward(a, in_back_propogation = True)
            # suppose one neuron in the next layer, with weights all 0.1, bias is zero, sigmoid and cross-entropy cost, y is zero
            y_ = Sigmoid.f(z.sum()-40)
            C = -np.log(1 - y_)  # suppose y = 0
            delta = y_
            d = self.conv_layer.back_propogation(delta * np.ones_like(z))
            return C, np.ravel(self.conv_layer.delta_w)[weight_index].copy()
        tiny_delta_weights = 0.0001
        #a = np.random.rand(1, 6, 6)
        a = np.array([0.74341247,  0.47463755,  0.00992929,  0.95730784,  0.20542349,
                        0.24606582,  0.5627104 ,  0.18438329,  0.89370057,  0.73840308,
                        0.96136674,  0.19538822,  0.1619067 ,  0.02462808,  0.85983933,
                        0.92236065,  0.87674389,  0.68733282,  0.4138197 ,  0.41656749,
                        0.38043692,  0.78814061,  0.30552122,  0.44576086,  0.79040761,
                        0.78019093,  0.95638804,  0.2221817 ,  0.18427876,  0.53748266,
                        0.23379542,  0.27326781,  0.14063543,  0.24078563,  0.54046106,
                        0.09593265]).reshape(1, 6, 6);
        cost1, delta_w1 = back_with_weights(a, tiny_delta_weights, weight_index = 1)
        cost2, delta_w2 = back_with_weights(a, -tiny_delta_weights, weight_index = 1)
        delta_w = (delta_w1 + delta_w2) /2
        gradient_w = (cost1 - cost2) / 2 / tiny_delta_weights
        Debug.print_('test_back_propogation:', 'delta_w1:', delta_w1, 'delta_w2:', delta_w2)
        Debug.print_('cost1:', cost1, 'cost2:', cost2)
        Debug.print_('gradient_w', gradient_w, 'delta_w:', delta_w)
        self.assertTrue(1)

if __name__ == '__main__':
    Debug.ENABLE = True
    unittest.main()