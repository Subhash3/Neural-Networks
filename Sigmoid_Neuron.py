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

    def activator(self, wx) :
        try :
            y = 1/(1 + math.exp(-wx))
        except OverflowError :
            wx = float(str(wx)[:3])
            y = 1/(1 + math.exp(-wx))
        return y


    def gradientDescent_epoch(self, data, size) :
        correct_guesses = 0
        loss = 0
        weight_change = [0]*(self.N+1)
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
            change_in_weight = self.LearningRate * error * prediction * (1 - prediction) * 1
            weight_change[0] += change_in_weight
            # self.Weights[0] += change_in_weight

            # Update W_0 to W_N
            # [-T, w1, w2, w3....]
            #     [x1, x2, x3....]
            for j in range(self.N) :
                change_in_weight = self.LearningRate * error * prediction * (1 - prediction) * inp[j]
                self.Weights[j+1] += change_in_weight
                weight_change[j+1] += change_in_weight

            loss += error**2

        # print(weight_change)
        for i in range(self.N+1) :
            self.Weights[0] += weight_change[i]

        self.T = -self.Weights[0]
        return correct_guesses, loss/2
