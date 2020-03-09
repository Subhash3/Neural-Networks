#!/usr/bin/python3

import csv
import sys
import random
from matplotlib import pyplot as plt
from p5 import *
import time

minX = maxX = None

class Perceptron() :
    N = None # Number of inputs
    Weights = list()
    T = None # Threshold
    LearningRate = 0.01
    def __init__(self, N) :
        self.N = N # initialize the number of inputs

        # Initialize all the weights to some random values
        for _ in range(N) :
            self.Weights.append(random.random()*random.randint(-2, 2) + random.random()*random.randint(-2, 2))
            # self.Weights.append(0)
        self.T = 1
        self.Weights.append(-self.T) # Corresponds to the input 1
    
    def guess(self, inp) :
        prediction = 0
        inp.append(1) # Corresponds to the weight -T
        prediction = self.weighted_sum(inp, self.Weights, self.N+1)
        # for i in range(self.N+1) :
        #     prediction += inp[i] * self.Weights[i]
        # inp.pop(-1) # Remove the last input that we added
        return self.activator(prediction)

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

    def train_epoch(self, data, size) :
        correct_guesses = 0
        for i in range(size) :
            # print("Weights:", self.Weights, "Threshold: ", self.T)
            sample = data[i]
            inp = sample[0]
            target = sample[1]
            prediction = self.guess(inp)

            error = target - prediction
            if error == 0 :
                correct_guesses += 1
            # print(target, prediction, error)
            
            for j in range(self.N+1) :
                self.Weights[j] += (inp[j] * self.LearningRate * error)
        return correct_guesses

    def train(self, data, size) :
        epochs = 10000
        for e in range(epochs) :
        # while True :
            # print("Epoch: ", e+1)
            # i = random.randint(0, size-1)
            
            correct_guesses = self.train_epoch(data, size)
            if correct_guesses == size :
                print("Successfully Trained by ", e, "th Epoch", sep='')
                return True
        print(epochs, "Epochs Exhausted..!!..Not sure if it has learned or not.!")
        return False

def check_usage() :
    argv = sys.argv
    argc = len(argv)

    if argc != 2 :
        print("Usage: ./classification.py <csv Dataset>")
        quit()

    data_file = argv[1]
    try :
        data_file_handler = open(data_file) # try to open file
    except Exception as e :
        print("Exception Occurred!: ", e)
        quit()
    return data_file_handler

def parse_data(data_file_handler, N) :
    global minX, maxX

    data = csv.reader(data_file_handler, delimiter=',')
    training_samples = list()  # List of tuples.. (input, output), again that input is a list
    training_size = 0

    for row in data :
        inp = list()

        if minX == None or float(row[0]) < minX :
            minX = float(row[0])
        if maxX == None or float(row[0]) > maxX :
            maxX = float(row[0])
        for i in range(N) :
            inp.append(float(row[i]))

        # if row[2] == ' Fighter' :
        #     # c = '#00FF00' # Plot fighter in green color
        #     target = 0
        # else :
        #     # c = '#FF0000' # plot bomber in red color
        #     target = 1
        # # target is 0 for fighter and 1 for bomber
        # data_sample = (inp, target)

        data_sample = (inp, float(row[N]))
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
    data_fp = check_usage()
    N = 2
    training_samples, training_size = parse_data(data_fp, N)
    perceptron = Perceptron(N)

    print("Weights:", perceptron.Weights, "\nThreshold: ", perceptron.T)
    for inp, target in training_samples :
        out = perceptron.guess(inp)
        print(inp[:N], out, target)
    plot_data(training_samples, training_size)

    perceptron.train(training_samples, training_size)

    print("Weights:", perceptron.Weights, "\nThreshold: ", perceptron.T)
    for inp, target in training_samples :
        out = perceptron.guess(inp)
        print(inp[:N], out, target)
    w = perceptron.Weights
    m = -(w[0]/w[1])
    c = -(w[2]/w[1])
    draw_line(m, c)

    plt.show()

# Images of (x, y) w.r.t y = k
def map_point(x, y, k) :
    return x, 2*k-y

def unmap_point(x, y, k) :
    return map_point(x, y, k)

def setup() :
    global training_samples, training_size
    global perceptron
    global EPOCHS, TRAINED

    size(500, 600)
    # data_fp = check_usage()
    N = 2
    # training_samples, training_size = parse_data(data_fp, N)
    perceptron = Perceptron(N)
    # print("Weights:", perceptron.Weights, "\nThreshold: ", perceptron.T)
    # for inp, target in training_samples :
    #     out = perceptron.guess(inp)
    #     print(inp[:N], out, target)     

def draw() :
    global FINISHED_COLLECTING
    global training_samples, training_size
    global minX, maxX
    global perceptron
    global EPOCHS, TRAINED

    background(50, 50, 50)

    for sample in training_samples :
            inp = sample[0]
            out = sample[1]

            # a, b = map_point(inp[0], inp[1], height/2)
            a = inp[0]
            b = inp[1]
            
            stroke(0)
            if out == 1 : # Bomber
                # color = "#009900"
                fill("#00aa00")
                circle((a, b), 8)
            else :
                # color = "#FF0000"
                fill("#aa0000")
                square((a, b), 8)
            no_fill()
            # circle((a, b), 8)

    if FINISHED_COLLECTING :
        w = perceptron.Weights
        m = -(w[0]/w[1])
        c = -(w[2]/w[1])

        x  = [minX, maxX]
        y = [m*i +c for i in x]
        
        if not TRAINED :
            print("Epoch:", EPOCHS)
            print("Weights: ", w)
            correct_guesses = perceptron.train_epoch(training_samples, training_size)
            EPOCHS += 1

            if correct_guesses == training_size :
                TRAINED = True
                print("Trained after ", EPOCHS, "Epochs")

        stroke(255)
        # line(map_point(x[0], y[0], height/2), map_point(x[1], y[1], height/2))
        line((x[0], y[0]), (x[1], y[1]))
        stroke(0)

def key_pressed(e) :
    global FINISHED_COLLECTING, LABEL, fp
    k = str(e.key)
    print(k)
    type(k)
    if k == 't' :
        FINISHED_COLLECTING = True
        fp.close()
    else :
        LABEL = int(k)
        print("Label:", LABEL)

def mouse_pressed(e) :
    # print(mouse_x, mouse_y)
    global FINISHED_COLLECTING, LABEL
    global training_samples, training_size
    global minX, maxX, fp

    if minX == None or mouse_x < minX :
        minX = float(mouse_x)
    if maxX == None or mouse_x > maxX :
        maxX = float(mouse_x)
    
    if not FINISHED_COLLECTING :
        # mapped_point = map_point(mouse_x, mouse_y, height/2)
        # training_samples.append(([mapped_point[0], mapped_point[1]], float(LABEL)))
        training_samples.append(([mouse_x, mouse_y], float(LABEL)))
        string = str(mouse_x) + ", " + str(mouse_y) + ", " + str(float(LABEL)) + "\n"
        fp.write(string)
        training_size += 1

if __name__ == "__main__":
    training_samples = list()
    training_size = 0
    perceptron = None
    EPOCHS = 0
    TRAINED = False
    LABEL = 1
    FINISHED_COLLECTING = False
    datasets_dir = "datasets/"
    filename = datasets_dir + str(time.time()).split('.')[0] + ".csv"
    fp = open(filename, 'w')
    run()