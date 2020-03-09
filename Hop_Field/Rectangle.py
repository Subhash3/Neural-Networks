#!/usr/bin/python3

import p5

class Rectangle() :
    def __init__(self, corner, width, height) :
        self.corner = corner
        self.height = height
        self.width = width
        self.color = 0
        p5.fill(self.color)

    def display(self) :
        x, y = self.corner
        p5.rect((x, y), self.width, self.height)
    
    def click(self) :
        if self.color == 0 :
            self.color = 255
        else :
            self.color = 0
        p5.fill(self.color)