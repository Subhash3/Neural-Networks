#!/usr/bin/python3

import random
import math

from Neuron import Neuron

class SigmoidNeuron(Neuron) :
    def weighted_sum(self, X, Y, l) :
        s = 0
        for i in range(l) :
            s += X[i] * Y[i]
        return s

    def activator(self, wx, derivative=False) :
        if derivative :
            # print("out: ", wx, "Derivative: ", wx* (1-wx))
            return wx * (1 - wx)

        try :
            y = 1/(1 + math.exp(-wx))
        except OverflowError :
            wx = float(str(wx)[:3])
            y = 1/(1 + math.exp(-wx))
        return y
