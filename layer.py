#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np

class Layer:
    def __init__(self, weight, bias, activation_function):
        self.activation = None
        self.weight = np.array(weight)
        self.bias = np.array(bias)
        self.activation_function = activation_function
        self.z = []

    def update_layer(self, activation, keep_z=False):
        z = np.dot(self.weight, activation) + self.bias
        self.activation = self.activation_function(z)
        if keep_z == True:
            self.z = z
        return self.activation

def create_layers(sizes):
    layers = []
    weights = []
    biases = []
    for s in sizes:
        weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
        biases = [np.random.randn(y,1) for y in sizes[1:]]
        layer = Layer(weights,biases, None)#o terceiro argumento é a função de ativação, por enquanto ta None pq a gnt ta perdido
        layers.append(layer)
    return layers
