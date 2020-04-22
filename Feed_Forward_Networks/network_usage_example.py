#!/usr/bin/python3

from Neural_Network import NeuralNetwork

brain = NeuralNetwork(2, 2, 1, [2, 2])
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

brain.Train(XOR_data, size, graph=True, MAX_EPOCHS=6000)
brain.print_weights()
inp = [0, 0]
output = brain.predict(inp)
print("\nInput: ", inp, "Prediction: ", output)
inp = [0, 1]
output = brain.predict(inp)
print("Input: ", inp, "Prediction: ", output)
inp = [1, 0]
output = brain.predict(inp)
print("Input: ", inp, "Prediction: ", output)
inp = [1, 1]
output = brain.predict(inp)
print("Input: ", inp, "Prediction: ", output)
brain.evaluate()