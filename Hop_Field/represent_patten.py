#!/usr/bin/python3

from p5 import *
import sys

grid = list()
w = 20 # width of each cell
data_fp = None
data_vector = list()
N = 10

def open_file(filename, mode='r') :
    try :
        fp = open(filename, mode=mode)
    except Exception as e :
        print("Error: ", e)
        quit()
    
    return fp

def get_file() :
    argv = sys.argv
    argc = len(argv)

    if argc != 2 :
        print("Usage: ", argv[0], "<Data File>")
        quit()

    data_fp = open_file(argv[1])

    return data_fp

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
    global data_fp, data_vector
    size(200,200)

    data_fp = get_file()
    data = parse_file(data_fp)

    data_vector = data[0]
    no_loop()
    
def draw():
    global data_fp, data_vector
    x,y = 0,0 # starting position

    create_grid(data_vector)
    print_grid()

    for row in grid:
        for col in row:
          if col == 1:
              fill(150,50,90)
          else:
              fill(255)
          rect((x, y), w, w)
          x = x + w  # move right
        y = y + w #  move down
        x = 0 # rest to left edge

if __name__ == "__main__":
    run()