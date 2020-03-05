#!/usr/bin/python3

from Matrix import Matrix
import numpy

a = Matrix(3, 2)
b = Matrix(2, 3)

a.randomize()
b.randomize()

print("A: ")
a.display()
print("B: ")
b.display()

print("C: ")
c = a.transpose()
c.display()

print("A: ")
a.add(b)
a.display()

d = a.multiply(b)
a.display()
b.display()
c.display()
d.display()
print(numpy.dot(a.table, b.table))