#! /usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from time import sleep

def plot(y,x):
    plt.plot(y,x)
    plt.xlabel("Epoch")
    plt.ylabel("Hit rate")
    plt.title("Results")
    plt.show()
