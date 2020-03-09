#!/usr/bin/python3

from Perceptron import Perceptron

class NeuralNetwork() :
    input_layer = list()
    output_layer = list()

    def __init__(self, I, H, O, list_of_hidden_nodes=None) :

        if list_of_hidden_nodes != None and H != len(list_of_hidden_nodes) :
            # print("Invalid Arguments.! NeuralNetwork Object must have (No. of inputs), (No. of Hidden Layers), (No. of outputs), (List containing the No. of nodes in each of the hidden layer)")
            print("Length of Hidden Nodes list{", len(list_of_hidden_nodes), "}, and H{", H,  "} must be same", sep="")
            return None

        self.I = I # No. of inputs
        self.H = H # No. of Hidden Layers
        self.O = O # No. of Outputs
        self.HiddenLayers = list()

        for _ in range(I) :
            self.input_layer.append(0) # Assume that all the inputs are zero
        
        for i in range(H) :
            layer = list()
            if i == 0 :
                inputs = I # First Hidden layer is connected to Input layer
            else : 
                inputs = list_of_hidden_nodes[i-1] # Rest of the hidden layers are connected to its previous hidden layer

            neuron = Perceptron(inputs)
            for _ in range(list_of_hidden_nodes[i]) :
                layer.append(neuron)

            self.HiddenLayers.append(layer)

        if H == 0 :
            inputs = self.I
        else :
            inputs = list_of_hidden_nodes[-1]

        neuron = Perceptron(inputs)
        for i in range(O) :
            self.output_layer.append(neuron)

    def print_weights(self) :
        for h in range(self.H) :
            layer = self.HiddenLayers[h]
            print("Hidden Layer -", h+1)

            i = 1
            for neuron in layer :
                print("\t Neuron: ", i, end="\n\t")
                print("Weights: ", neuron.Weights, len(neuron.Weights))
                i += 1

        i = 1
        print("Output Layer")
        for neuron in self.output_layer :
            print("\t Neuron: ", i, end="\n\t")
            print("Weights: ", neuron.Weights, len(neuron.Weights))
            i += 1
            

    def feedForward(self, inp) :
        for h in range(self.H) :
            output = list()
            hiddden_layer = self.HiddenLayers[h]
            for neuron in hiddden_layer :
                prediction = neuron.guess(inp)
                output.append(prediction)
            inp = output

        output = list()
        for neuron in self.output_layer :
            prediction = neuron.guess(inp)
            output.append(prediction)

        return output

    
    def backpropagate(self, out, target) :
        weight_changes = list()
        for i in range(self.O) :
            neuron = self.output_layer[i]
            dw = [0]*self.O
            for j in range(neuron.N) :
                if self.H == 0 :
                    In = self.input_layer[j]
                else :
                    In = self.HiddenLayers[-1][j]
                for k in range(self.O) :
                    dw[k] += (target[k] - out[k]) * out[k] * (1 - out[k]) * In
            weight_changes.append(dw)
        return weight_changes

    def Train(self, data, size) :
        for i in range(size) :
            sample = data[i]
            inp = sample[0]
            target = sample[1]

            out = self.feedForward(inp)
            print(sample, out, target)
            weight_changes = self.backpropagate(out, target)