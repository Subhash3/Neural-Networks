#!/usr/bin/python3

from Sigmoid_Neuron import SigmoidNeuron as MyNeuron
import pickle
from matplotlib import pyplot as plt

class NeuralNetwork() :
    input_layer = list()
    output_layer = list()

    def __init__(self, I, H, O, list_of_hidden_nodes=None) :
        self.LearningRate = 0.7

        if list_of_hidden_nodes != None and H != len(list_of_hidden_nodes) :
            # print("Invalid Arguments.! NeuralNetwork Object must have (No. of inputs), (No. of Hidden Layers), (No. of outputs), (List containing the No. of nodes in each of the hidden layer)")
            print("Length of Hidden Nodes list{", len(list_of_hidden_nodes), "}, and H{", H,  "} must be same", sep="")
            return None

        self.I = I # No. of inputs
        self.H = H # No. of Hidden Layers
        self.O = O # No. of Outputs
        self.HiddenLayers = list()
        self.list_of_hidden_nodes = list_of_hidden_nodes
        self.MSE = 0 # Mean Squared Error

        for _ in range(I) :
            self.input_layer.append(0) # Assume that all the inputs are zero
        
        for i in range(H) :
            layer = list()
            if i == 0 :
                inputs = I # First Hidden layer is connected to Input layer
            else : 
                inputs = list_of_hidden_nodes[i-1] # Rest of the hidden layers are connected to its previous hidden layer

            for _ in range(list_of_hidden_nodes[i]) :
                neuron = MyNeuron(inputs)
                layer.append(neuron)

            self.HiddenLayers.append(layer)

        if H == 0 :
            inputs = self.I
        else :
            inputs = list_of_hidden_nodes[-1]

        for i in range(O) :
            neuron = MyNeuron(inputs)
            self.output_layer.append(neuron)

    def print_weights(self) :
        print("Input Layer: ", self.input_layer)
        for h in range(self.H) :
            layer = self.HiddenLayers[h]
            print("Hidden Layer -", h+1)

            i = 1
            for neuron in layer :
                print("\t Neuron: ", i, end="\n\t")
                print("Weights: ", neuron.Weights, len(neuron.Weights), end="\n\t")
                print("Delta: ", neuron.delta, end="\n\t")
                print("Output: ", neuron.predicted_value)
                i += 1

        i = 1
        print("Output Layer")
        for neuron in self.output_layer :
            print("\t Neuron: ", i, end="\n\t")
            print("Weights: ", neuron.Weights, len(neuron.Weights), end="\n\t")
            print("Delta: ", neuron.delta, end="\n\t")
            print("Output: ", neuron.predicted_value)
            i += 1
            

    def feedForward(self, inp) :
        # print("Feed Forward")
        for h in range(self.H) :
            output = list()
            hiddden_layer = self.HiddenLayers[h]
            for neuron in hiddden_layer :
                prediction = neuron.guess(inp)
                neuron.predicted_value = prediction
                output.append(prediction)
            inp = output

        output = list()
        for neuron in self.output_layer :
            prediction = neuron.guess(inp)
            neuron.predicted_value = prediction
            output.append(prediction)

        return output

    def backpropagate(self, target) :
        # print("Back Propagation")
        total_error = 0 # Mean Squared Error

        # Calculate the delta of output layer neurons
        output_errors = list()
        for j in range(self.O) :
            # Error in output layer: (target - output) * activation_derivative(output)
            neuron = self.output_layer[j]
            error = target[j] - neuron.predicted_value
            total_error += error # calculate total error

            neuron.delta = error * neuron.activator(neuron.predicted_value, derivative=True)
            # neuron.delta = error * out[j] * (1 - out[j]) # for sigmoid activator
            output_errors.append(error)

        # Calculate the delta of hidden layers
        for h in range(self.H-1, -1, -1) :
            hiddden_layer = self.HiddenLayers[h]
            num_nodes = self.list_of_hidden_nodes[h]
            for j in range(num_nodes) :
                error_in_this_particular_hidden_neuron = 0
                if h == self.H-1 :
                    # this hidden layer is connected to output layer
                    next_layer = self.output_layer
                else :
                    next_layer = self.HiddenLayers[h+1]

                for neuron in next_layer :
                    error_in_this_particular_hidden_neuron += (neuron.Weights[j+1] * neuron.delta)

                hidden_neuron = hiddden_layer[j]
                output_from_this_neuron = hidden_neuron.predicted_value
                hidden_neuron.delta = error_in_this_particular_hidden_neuron * hidden_neuron.activator(output_from_this_neuron, derivative=True)

        return output_errors, total_error

    def update_weights(self) :
        # print("Updating Weights")
        # Update weights of the output layer neurons
        if self.H == 0 :
            # No hidden layers to update
            for neuron in self.output_layer :
                neuron.Weights[0] += self.LearningRate * neuron.delta * 1 # Bias
                for k in range(self.I) :
                    neuron.Weights[k+1] += self.LearningRate * neuron.delta * self.input_layer[k]
        else :
            last_hidden_layer = self.HiddenLayers[self.H-1]
            nodes_in_last_hidden_layer = self.list_of_hidden_nodes[self.H-1]
            for neuron in self.output_layer :
                neuron.Weights[0] += self.LearningRate * neuron.delta * 1 # Bias
                for k in range(nodes_in_last_hidden_layer) :
                    neuron.Weights[k+1] += self.LearningRate * neuron.delta * last_hidden_layer[k].predicted_value

            # Update weights of hidden layer neurons
            # leave the first hidden layer, as we have to update it using input layer
            if self.H > 1 :
                for h in range(self.H-1, 0, -1) :
                    hiddden_layer = self.HiddenLayers[h]
                    current_num_nodes = self.list_of_hidden_nodes[h]
                    previous_hidden_layer = self.HiddenLayers[h-1]
                    previous_nodes = self.list_of_hidden_nodes[h-1]
                    for j in range(current_num_nodes) :
                        current_hidden_neuron = hiddden_layer[j]
                        for k in range(previous_nodes) :
                            prev_hidden_neuron = previous_hidden_layer[k]
                            current_hidden_neuron.Weights[k+1] += self.LearningRate * current_hidden_neuron.delta * prev_hidden_neuron.predicted_value
                        # Update Bias
                        current_hidden_neuron.Weights[0] += self.LearningRate * current_hidden_neuron.delta * 1

            # Update the weights of first hidden layer
            hiddden_layer = self.HiddenLayers[0]
            current_num_nodes = self.list_of_hidden_nodes[0]
            for j in range(current_num_nodes) :
                current_hidden_neuron = hiddden_layer[j]
                for k in range(self.I) :
                    current_hidden_neuron.Weights[k+1] += self.LearningRate * current_hidden_neuron.delta * self.input_layer[k]
                # Update Bias
                current_hidden_neuron.Weights[0] += self.LearningRate * current_hidden_neuron.delta * 1



    def Train(self, data, size, MAX_EPOCHS = 10000, graph=False) :
        all_erros = list()
        for epoch in range(MAX_EPOCHS) :
            # print("Epoch:", epoch+1)
            print("Epoch: ", epoch+1, "Error: ", self.MSE)
            self.MSE = 0 
            for i in range(size) :
                sample = data[i]
                inp = sample[0]
                target = sample[1]
                self.input_layer = inp

                # Feed forward the input
                out = self.feedForward(inp)
                # Backpropagate the error
                out_errors, total_error = self.backpropagate(target)
                # Update weights
                self.update_weights()

                print(sample, out, target, out_errors)
                # self.print_weights()
                self.MSE += (total_error * total_error)
            all_erros.append(self.MSE)
                
            print()
        if graph :
            self.epoch_vs_error(all_erros, MAX_EPOCHS)
            
    def predict(self, inputs) :
        output = self.feedForward(inputs)
        return output

    def epoch_vs_error(self, all_erros, epochs) :
        plt.title("Epoch vs Error")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        all_epochs = [i+1 for i in range(epochs)]

        plt.plot(all_epochs, all_erros)
        plt.show()

    def export_network(self, filename) :
        try :
            file_to_write = open(filename, 'wb')
        except Exception as e :
            print("Exception Occurred While opening file: ", filename, ": ", e)
            print("Couldn't export model")
            return
        pickle.dump(self, file_to_write)
        file_to_write.close()

    @staticmethod
    def load_network(filename) :
        try :
            file_to_read = open(filename, 'rb')
        except Exception as e :
            print("Exception Occurred While opening file: ", filename, ": ", e)
            print("Couldn't Load model")
            return None
        network = pickle.load(file_to_read)
        file_to_read.close()

        return network