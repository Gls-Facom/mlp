#! /usr/bin/python
# -*- coding: utf-8 -*-
import sys
import activationFunctions as af

def load_config():
    f = open(sys.argv[1],"r")
    file = []
    for line in f:
        file.append(line)
    line = file[0]
    sizes = [int(s) for s in line.split(",")]
    activation_name = file[-1]
    file = file[1:-1]
    args = []
    for line in file:
        args.append(float(line))
    if activation_name == "relu":
        activation = af.relu
    elif activation_name == "tanh":
        activation = af.tanh
    elif activation_name == "leaky_relu":
        activation = af.leaky_relu
    else:
        activation = af.sigmoid


    return sizes, args[0], args[1], args[2], activation
