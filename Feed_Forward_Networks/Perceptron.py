#!/usr/bin/python3

import random
import math

from Neuron import Neuron

class Perceptron(Neuron) :
    def weighted_sum(self, X, Y, l) :
        s = 0
        for i in range(l) :
            s += X[i] * Y[i]
        return s

    def activator(self, wx) :
        if wx >= 0 :
            return 1
        else :
            return 0