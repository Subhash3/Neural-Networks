#!/usr/bin/python3

from p5 import *
import sys
from random import randint

from Hop_Field import HopFieldNetwork

argv = sys.argv
argc = len(argv)

if argc != 3 :
    print("Usage: ./test_hop_field.py <training> <testing>")
    quit()

N = 10

steps = 0
training_data = list()
testing_data = list()
testing_vector = list()
prev_inp = list()
brain = HopFieldNetwork(N**2)

i = 0

grid = list()
w = 20 # width of each cell
def open_file(filename, mode='r') :
    try :
        fp = open(filename, mode=mode)
    except Exception as e :
        print("Error: ", e)
        quit()
    
    return fp

def get_files() :
    argv = sys.argv
    argc = len(argv)

    if argc != 3 :
        print("Usage: ", argv[0], "<Training Data> <Testing Data>")
        quit()

    training_file = open_file(argv[1])
    testing_file = open_file(argv[2])

    return training_file, testing_file

def parse_file(fp) :
    file_data = fp.readlines()
    data = list()

    for line in file_data :
        row = list(map(int, line.split()))
        data.append(row)
    fp.close()
    return data

def print_grid() :
    for row in grid :
        print(row)

def create_grid(arr) :
    global grid, N
    grid = list()
    # print(arr)
    for i in range(N) :
        grid.append(arr[i*N:i*N + N])
    print()

def setup():
    global brain, training_data, testing_data, testing_vector

    size(200,200)

    training_fp, testing_fp = get_files()
    training_data = parse_file(training_fp)
    testing_data = parse_file(testing_fp)

    for row in training_data :
        brain.learn_input(row)

    testing_vector = testing_data[0]
    
def draw():
    global brain, testing_data, grid, testing_vector, prev_inp, steps, i
    x,y = 0,0 # starting position
    testing_vector = brain.predict_one_val(testing_vector, i)
    steps += 1

    create_grid(testing_vector)
    print_grid()

    for row in grid:
        for col in row:
          if col == 1:
              fill(150,50,90)
          else:
              fill(255)
          rect((x, y), w, w)
          x = x + w  # move right
        y = y + w # move down
        x = 0 # rest to left edge

    if i == brain.N -1 :
        i = 0
        try :
            if all(prev_inp == testing_vector) : # All elements of two arrays are same
                print("Took ", steps, "Steps to converge")
                no_loop()
        except :
            pass
        prev_inp = testing_vector  
    i += 1

if __name__ == "__main__":
    run()
