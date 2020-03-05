#!/usr/bin/python3

from Neural_Network import NeuralNetwork

brain = NeuralNetwork(2, 2, 1, [3, 2])
inp = [0, 0]
out = brain.feedForward(inp)
print(out)
inp = [0, 1]
out = brain.feedForward(inp)
print(out)
inp = [1, 0]
out = brain.feedForward(inp)
print(out)
inp = [1, 1]
out = brain.feedForward(inp)
print(out)

brain.print_weights()