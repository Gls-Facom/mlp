#!/usr/bin/python

import numpy as np

class Layer:
    def __init__(self, weight, bias, activation_function):
        self.activation = None
        self.weight = weight
        self.bias = bias
        self.activation_function = activation_function
        self.z = []

    def update_layer(self, activation, keep_z=False):
        z = np.dot(self.weight, activation) + self.bias
        self.activation = self.activation_function(z)
        if keep_z == True:
            self.z = z
        return self.activation
