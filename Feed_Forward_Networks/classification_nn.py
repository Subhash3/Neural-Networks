#!/usr/bin/python3

import csv
import sys
import random
from matplotlib import pyplot as plt
import numpy as np
from Perceptron import Perceptron

minX = maxX = None

def check_usage() :
    argv = sys.argv
    argc = len(argv)

    if argc != 3 :
        print("Usage: ./classification.py <Training Dataset (CSV)> <Testing Dataset (CSV)>")
        quit()

    training_data_file = argv[1]
    try :
        training_fp = open(training_data_file) # try to open file
    except Exception as e :
        print("Exception Occurred!: ", e)
        quit()

    testing_data_file = argv[2]
    try :
        testing_fp = open(testing_data_file) # try to open file
    except Exception as e :
        print("Exception Occurred!: ", e)
        quit()

    return training_fp, testing_fp

def parse_data(data_file_handler, N) :
    global minX, maxX

    data = csv.reader(data_file_handler, delimiter=',')
    training_samples = list()  # List of tuples.. (input, output), again that input is a list
    training_size = 0

    for row in data :
        inp = list()
        _out = list()

        if minX == None or float(row[0]) < minX :
            minX = float(row[0])
        if maxX == None or float(row[0]) > maxX :
            maxX = float(row[0])
        for i in range(N) :
            inp.append(float(row[i]))

        data_sample = [inp, float(row[N])]
        training_samples.append(data_sample)
        training_size += 1

    return training_samples, training_size

def plot_data(data, size) :
    for i in range(size) :
        sample = data[i]
        inp = sample[0]
        out = sample[1]

        if out == 1 : # Bomber
            color = "#009900"
        else :
            color = "#FF0000"
        plt.scatter(*inp, c=color)
    return

def draw_line(m, c) :
    global minX, maxX

    x  = [minX, maxX]
    y = [m*i +c for i in x]
    plt.plot(x, y)
    return

def main() :
    training_fp, testing_fp = check_usage()
    N = 2
    training_samples, training_size = parse_data(training_fp, N)
    brain = Perceptron(N)

    print("Weights:", brain.Weights, "\nThreshold: ", brain.T)
    for inp, target in training_samples :
        out = brain.guess(inp)
        print(inp[:N], out, target, target - out)

    plt.figure(1)
    plt.grid()
    plot_data(training_samples, training_size)
    plt.title("Training Dataset")
    plt.xlabel("Mass")
    plt.ylabel("Speed")

    _epochs, all_errors = brain.train(training_samples, training_size)
    # if epochs == -1 :
    #     print(brain.epochs_taken, "Epochs Exhausted..!!..Not sure if it has learned or not.!")
    # else :
    #     print("Successfully Trained by ", epochs, "th Epoch", sep='')

    print()
    print("\tTraining completed (", brain.epochs_taken, " Epochs Taken)", sep="")
    print("\tConfidence: ", str(brain.confidence)[:5], "%", sep="")
    print()

    plt.figure(2)
    plt.grid()
    plt.plot([e for e in range(brain.epochs_taken)], all_errors)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Epoch vs Error")
    # plt.show()

    print("Weights:", brain.Weights, "\nThreshold: ", brain.T)
    for inp, target in training_samples :
        out = brain.guess(inp)
        print(inp[:N], out, target, target - out)

    m, c = brain.fit_hyperplane()
    # plt.subplot(1, 3, 1)
    plt.figure(1)
    draw_line(m, c)

    print("Testing....")
    testing_samples, testing_size = parse_data(testing_fp, N)
    for i in range(testing_size) :
        sample = testing_samples[i]
        inp = sample[0]
        target = sample[1]

        out = brain.guess(inp)
        testing_samples[i][1] = np.round(out)
        print(inp, target, out)

    plt.figure(3)
    plt.title("Testing Dataset")
    plot_data(testing_samples, testing_size)
    draw_line(m, c)
    plt.show()

if __name__ == "__main__":
    main()