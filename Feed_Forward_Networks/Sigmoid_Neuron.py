#!/usr/bin/python3

import random
import math

from Neuron import Neuron

class SigmoidNeuron(Neuron) :
    def weighted_sum(self, X, Y, l) :
        """
        Calculates the weighted sum of input

        Parameters
        ----------
        X : list
            Input vector
        Y : list
            Weights of the neuron including bias
        l : int
            length of the input vector (== weights vector)

        Returns
        -------
        s : float
            Weighted sum of the inputs
        """
        s = 0
        for i in range(l) :
            s += X[i] * Y[i]
        return s

    def activator(self, wx, derivative=False) :
        """
        Activation function of the neuron

        Parameters
        ----------
        wx : float
            Weighted sum of the input
        [derivative] : bool
            It is an optional parameter.
            If it is true, the derivative of the activation function is returned
        
        Returns
        -------
            : float
            If derivative = True, The derivative of the activation function is returned
            Otherwise the sigmoid ouptut will be returned
        """
        if derivative :
            # print("out: ", wx, "Derivative: ", wx* (1-wx))
            return wx * (1 - wx)

        try :
            y = 1/(1 + math.exp(-wx))
        except OverflowError :
            wx = float(str(wx)[:3])
            y = 1/(1 + math.exp(-wx))
        return y
