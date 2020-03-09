#!/usr/bin/python3

import numpy as np

class HopFieldNetwork() :
    def __init__(self, N) :
        self.N = N
        self.weight_matrix = np.zeros((N, N))
        self.isNormalised = False

    def learn_input(self, X) :
        X = np.array(X)
        X = X.reshape(self.N, 1)
        X_transpose = X.transpose()
        self.weight_matrix += X.dot(X_transpose)
        self.make_diag_zero()

    def make_diag_zero(self) :
        for i in range(self.N) :
            self.weight_matrix[i, i] = 0

    def normalise(self) :
        if not self.isNormalised :
            self.weight_matrix /= self.N
            self.isNormalised = True

    def predict_one_val(self, X, i) :
        self.normalise()
        X = np.array(X)
        i_th_column_of_weight_matrix = self.weight_matrix[:, i]
        net = X.dot(i_th_column_of_weight_matrix)
        val = X[i]
        X[i] = self.activation(net, val)

        return X

    def predict_one_step(self, X) :
        # print("LENGHT: ", len(X), self.N)
        for i in range(self.N) :
            X = self.predict_one_val(X, i)
        return X
    
    def predict(self, X) :
        prev_inp = None
        self.steps = 0
        # print("HAHA: ", len(X), self.N)
        while True :
            X = self.predict_one_step(X)
            self.steps += 1
            if all(prev_inp == X) : # All elements of two arrays are same
                return X
            prev_inp = X

    def activation(self, net, val) :
        if net > 0 :
            return 1
        elif net == 0 :
            return val
        else :
            return -1