#!/usr/bin/python3

import random
import math

from abc import abstractmethod
# abc = Abstract Base Class.
# abc is an inbuilt python module which allows you to create 
# Abstract classes and methods

class Neuron() :
    def __init__(self, N) :
        """
        Neuron class constructor: Creates a Neuron object.

        Parameters
        ----------
        N : Integer
            Number of inputs to the Neuron

        Returns
        -------
        Neuron
            Returns Neuron Object
        """
        Weights = list()
        self.N = N # initialize the number of inputs
        # self.T = random.random()*random.randint(-2, 2) + random.random()*random.randint(-2, 2)
        # self.Weights.append(-self.T) # Corresponds to the input '1'

        # Initialize all the weights to some random values
        for _ in range(N+1) :
            Weights.append(random.random()*random.randint(-2, 2) + random.random()*random.randint(-2, 2))
            # Weights.append(0)

        self.T = -Weights[0]
        self.Weights = Weights
        # self.LearningRate = 0.5
        self.epochs_taken = 0
        self.confidence = 0
        self.delta = 0
        self.predicted_value = 0

    def guess(self, inp) :
        """
        Makes a guess based on the given input

        Parameters
        ----------
        inp : list
            Input vector

        Returns
        -------
        prediction : float
            Output predicted by the neuron
        """
        prediction = 0
        inp.insert(0, 1) # Corresponds to the weight -T
        # print("Input:", inp, "Weights:", self.Weights)
        prediction = self.weighted_sum(inp, self.Weights, self.N+1)

        inp.pop(0) # Remove the first input that we added
        return self.activator(prediction)

    @abstractmethod
    def weighted_sum(self, X, Y, l) :
        pass
    # Aggregate function, differs from neuron to neuron
    # and so is activation function

    @abstractmethod
    def activator(self, wx, derivative=True) :
        pass
