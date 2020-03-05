#!/usr/bin/python3

import random
import math

from abc import abstractmethod
# abc = Abstract Base Class.
# abc is an inbuilt python module which allows you to create 
# Abstract classes and methods

class Neuron() :
    def __init__(self, N) :
        Weights = list()
        self.N = N # initialize the number of inputs
        self.T = random.random()*random.randint(-2, 2) + random.random()*random.randint(-2, 2)
        # self.Weights.append(-self.T) # Corresponds to the input '1'

        # Initialize all the weights to some random values
        for _ in range(N+1) :
            Weights.append(random.random()*random.randint(-2, 2) + random.random()*random.randint(-2, 2))
            # self.Weights.append(0)

        self.T = -Weights[0]
        self.Weights = Weights
        self.LearningRate = 0.1
        self.epochs_taken = 0
        self.confidence = 0

    def guess(self, inp) :
        prediction = 0
        inp.insert(0, 1) # Corresponds to the weight -T
        prediction = self.weighted_sum(inp, self.Weights, self.N+1)

        inp.pop(0) # Remove the first input that we added
        return self.activator(prediction)

    @abstractmethod
    def weighted_sum(self, X, Y, l) :
        pass
    # Aggregate function, differs from neuron to neuron
    # So is activation function

    @abstractmethod
    def activator(self, wx) :
        pass

    @abstractmethod
    def gradientDescent_epoch(self, data, size) :
        pass

    def train_epoch(self, data, size) :
        correct_guesses = 0
        loss = 0
        for i in range(size) :
            sample = data[i]
            inp = sample[0]
            target = sample[1]

            # Get the prdicted output for a given input
            prediction = self.guess(inp)

            error = target - prediction
            if error == 0 :
                correct_guesses += 1
            
            # update W_0 ==> Threshold which corresponds to the input 1
            change_in_weight = 1 * self.LearningRate * error
            self.Weights[0] += change_in_weight
            # loss += change_in_weight**2

            # Update W_0 to W_N
            # [-T, w1, w2, w3....]
            #     [x1, x2, x3....]
            for j in range(self.N) :
                change_in_weight = inp[j] * self.LearningRate * error
                self.Weights[j+1] += change_in_weight
                # loss += change_in_weight**2
            loss += abs(error)
        self.T = -self.Weights[0]
        return correct_guesses, loss

    def train(self, data, size, gradientDescent=False) :
        MAX_EPOCHS = 100000
        all_errors = list()
        loss = None
        for e in range(MAX_EPOCHS) :
            # if e%100 == 0 :
            #     print("Epoch: ", e, "Weights: ", self.Weights)
            #     print("Error: ", loss)
            # print("Weights: ", self.Weights, "Epoch: ", e)
            if gradientDescent :
                correct_guesses, loss = self.gradientDescent_epoch(data, size)
            else :         
                correct_guesses, loss = self.train_epoch(data, size)
            all_errors.append(loss)


            if loss <= 0.01 :
                self.epochs_taken = e+1
                self.confidence = (1-loss) *100
                return e+1, all_errors

            if correct_guesses == size : # it is never true in case of a sigmoid neuron
                self.epochs_taken = e+1
                self.confidence = (1-loss) *100
                return e+1, all_errors

        self.epochs_taken = e+1
        self.confidence = (1-loss) *100
        return -1, all_errors

    def fit_hyperplane(self) :
        w = self.Weights
        m = -(w[1]/w[2])
        c = -(w[0]/w[2])

        return m, c