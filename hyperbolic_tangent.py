#!/usr/bin/python3

import math
from matplotlib import pyplot as plt

def f(x) :
    return (1 - math.exp(-x))/(1 + math.exp(-x))

def df(x) :
    return (1-f(x)**2)

x = list()
y = list()
dy = list()
A = 10
i = -A
while i < A+1 :
    x.append(i)
    y.append(f(i))
    dy.append(df(i))
    i += 0.1

plt.figure(1)
plt.grid()
plt.plot(x, y)

plt.figure(2)
plt.grid()
plt.plot(x, dy)

plt.show()