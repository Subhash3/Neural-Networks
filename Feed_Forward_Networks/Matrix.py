#!/usr/bin/python3

import random

class Matrix() :
    def __init__(self, rows, cols) :
        self.rows = rows
        self.cols = cols
        self.table = list()

        for _i in range(self.rows) :
            l = list()
            for _j in range(self.cols) :
                l.append(0)
            self.table.append(l)

    def randomize(self) :
        for i in range(self.rows) :
            for j in range(self.cols) :
                p = random.random()*10
                p = float(str(p)[:4])
                self.table[i][j] = p

    def display(self) :
        print()
        for row in self.table :
            # print(row)
            for num in row :
                print(num, end=", ")
            print()
        print()


    def transpose(self) :
        new_matrix = Matrix(self.cols, self.rows)
        for i in range(self.cols) :
            l = list()
            for j in range(self.rows) :
                l.append(self.table[j][i])
            new_matrix.table.append(l)
        return new_matrix

    def add(self, n) :
        if isinstance(n, Matrix) :
            # Provided value is a matrix
            if self.rows != n.rows or self.cols != n.cols :
                return False
            for i in range(self.rows) :
                for j in range(self.cols) :
                    self.table[i][j] += n.table[i][j]
        else : # n is an integer
            for i in range(self.rows) :
                for j in range(self.cols) :
                    self.table[i][j] += n
        return True

    def multiply(self, n) :
        if isinstance(n, Matrix) :
            # Matrix Multiplication
            if self.cols != n.rows :
                return None
            result = Matrix(self.rows, n.cols)
            for i in range(result.rows) :
                for j in range(result.cols) :
                    for k in range(self.cols) :
                        result.table[i][j] += self.table[i][k] * n.table[k][j]
            return result

        else : # n is an integer
            for i in range(self.rows) :
                for j in range(self.cols) :
                    self.table[i][j] *= n