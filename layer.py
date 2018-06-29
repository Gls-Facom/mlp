#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np

class Layer:
    def __init__(self, weight, bias, activation_function, size):
        self.activation = None
        self.weight = np.asarray(weight)
        self.bias = np.asarray(bias)
        self.activation_function = activation_function
        self.z = []
        self.size = size

    def update_layer(self, activation, keep_z=False):
        z = np.dot(self.weight, activation) + self.bias
        self.activation = self.activation_function(z, False)
        if keep_z == True:
            self.z = z
        return self.activation

def create_layers(sizes, activation_function):
    layers = []
    weights = []
    biases = []
    # gerar todos os pesos e biases e pegar um por um pra cada layer e fazer append na lista de layers
    weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
    biases = [np.random.randn(y,1) for y in sizes[1:]]
    layer = Layer(None, None, activation_function, sizes[0])
    layers.append(layer)
    for i in xrange(len(sizes[1:])):
        layer = Layer(weights[i],biases[i], activation_function, sizes[i])
        layers.append(layer)

    return layers
