#!/usr/bin/python3

from Neural_Network import NeuralNetwork

brain = NeuralNetwork(2, 1, 1, [2])
XOR_data = [
    [
        [0, 0],
        [0]
    ],
    [
        [0, 1],
        [1]
    ],
    [
        [1, 0],
        [1]
    ],
    [
        [1, 1],
        [0]
    ]
]

size = 4

brain.Train(XOR_data, size, graph=True)
brain.print_weights()
inp = [1, 1]
output = brain.predict([1, 1])
print("\nInput: ", inp, "Prediction: ", output)