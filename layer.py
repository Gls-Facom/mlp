#!/usr/bin/python
import numpy as np

class Layer():
    number_of_neurons = None
    neurons = []
    bias = []
    z = []
    def __init__(self, neurons):
        self.neurons = np.array(neurons)
        self.number_of_neurons = len(neurons)

    def set_z(self, z):
        self.z = z
        
    def set_bias(self, bias):
        self.bias = bias
