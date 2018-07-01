#! /usr/bin/python
# -*- coding: utf-8 -*-
import sys
import activationFunctions as af

def load_config():
    f = open(sys.argv[2],"r")
    file = []
    for line in f:
        file.append(line)
    line = file[0]
    sizes = [int(s) for s in line.split(",")]
    learning_path = None
    if len(file) > 3:
        activation_name = file[-2].split("\n")[0]
        learning_path = file[-1].split("\n")[0]
    else:
        activation_name = file[-2].split("\n")[0]
    args = file[1].split(",")
    epochs = int(args[0])
    mbs = int(args[1])
    eta = float(args[2])

    if activation_name == "relu":
        activation = af.relu
    elif activation_name == "tanh":
        activation = af.tanh
    elif activation_name == "leaky_relu":
        activation = af.leaky_relu
    else:
        activation = af.sigmoid
    if sys.argv[1] == "-train":
        learning_path = None
    return sizes, epochs, mbs, eta, activation, learning_path
