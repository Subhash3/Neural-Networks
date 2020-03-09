#!/usr/bin/python3

from p5 import *
import sys

argv = sys.argv
argc = len(argv)

if argc != 2 :
    print("Usage: ./collect_data.py <file_name>")
    quit()

input_file = argv[1]
input_file = open(input_file, 'w')

N = 10
grid = [ [-1]*N  for n in range(N)] # list comprehension

w = 20 # width of each cell
input_vector = list()
    
def print_grid() :
    for row in grid :
        print(row)

def collect_data() :
    global input_vector
    input_vector = list()
    for row in grid :
        for i in row :
            input_vector.append(i)

def setup():
    size(200,200)
    no_loop()
    
def draw():
    x,y = 0,0 # starting position
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
    # print_grid()
    collect_data()
    print(input_vector)

def key_pressed(e) :
    if e.key == 'q' :
        for i in input_vector :
            input_file.write(str(i) + " ")
        input_file.write('\n')
        exit()
        
def mouse_pressed():
    global w
    # print(int(mouse_x/w), int(mouse_y/w))
    x = int(mouse_x/w)
    y = int(mouse_y/w)
    grid[y][x] = -1 * grid[y][x]
    redraw()
    # integer division is good here!

if __name__ == "__main__":
    run()
