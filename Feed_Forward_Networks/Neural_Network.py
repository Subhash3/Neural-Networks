#!/usr/bin/python3

from Sigmoid_Neuron import SigmoidNeuron as MyNeuron
import pickle
from matplotlib import pyplot as plt

class NeuralNetwork() :
    input_layer = list()
    output_layer = list()

    def __init__(self, I, H, O, list_of_hidden_nodes=None) :
        """
        Neural Network constructor: Creates a Neural Network object.

        Parameters
        ----------
        I : Integer
            Number of inputs to the network
        H : Integer
            Number of hidden layers of the network
        O : Integer
            Number of neurons in the output layer
        list_of_hidden_nodes : list
            list of number of nuerons in each hidden layer.

        Returns
        -------
        NeuralNetwork
            Returns NeuralNetwork Object
        """
        self.LearningRate = 0.8

        if list_of_hidden_nodes != None and H != len(list_of_hidden_nodes) :
            # print("Invalid Arguments.! NeuralNetwork Object must have (No. of inputs), (No. of Hidden Layers), (No. of outputs), (List containing the No. of nodes in each of the hidden layer)")
            print("Length of Hidden Nodes list{", len(list_of_hidden_nodes), "}, and H{", H,  "} must be same", sep="")
            return None

        self.I = I # No. of inputs
        self.H = H # No. of Hidden Layers
        self.O = O # No. of Outputs
        self.HiddenLayers = list()
        self.list_of_hidden_nodes = list_of_hidden_nodes
        self.MSE = 1 # Mean Squared Error
        self.confidence = 0

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
        """
        A debug function to take a look at the weights of the network.

        Parameters
        ----------
        Doesn't accepts any parameter

        Returns
        -------
        Doesn't return anything
        """
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
        """
        Feedforwards the given input

        Parameters
        ----------
        inp : list
            Input to the network.

        Returns
        -------
        Returns a tuple
        
        output : list
            Output from the output layer
        hidden_outputs : list
            It is list of lists containing the output from each hidden layer
        """
        # print("Feed Forward")
        hidden_outputs = list()
        for h in range(self.H) :
            output = list()
            hiddden_layer = self.HiddenLayers[h]
            for neuron in hiddden_layer :
                prediction = neuron.guess(inp)
                prediction = round(prediction, 7)
                neuron.predicted_value = prediction
                output.append(prediction)
            hidden_outputs.append(output)
            inp = output

        output = list()
        for neuron in self.output_layer :
            prediction = neuron.guess(inp)
            prediction = round(prediction, 7)
            neuron.predicted_value = prediction
            output.append(prediction)

        return output, hidden_outputs

    def backpropagate(self, target) :
        """
        Backpropagate the error throughout the network

        Parameters
        ----------
        target : list
            The ground truth target that corresponds to the input thats fed.

        Returns
        -------
        Returns a tuple
        
        output_errors : list
            Error in the output layer
        hidden_errors : list
            It is list of lists containing the error or each hidden layer
        """
        # print("Back Propagation")
        total_error = 0 # Mean Squared Error

        # Calculate the delta of output layer neurons
        output_errors = list()
        for j in range(self.O) :
            # Error in output layer: (target - output) * activation_derivative(output)
            neuron = self.output_layer[j]
            error = target[j] - neuron.predicted_value
            total_error += (error * error) # calculate squared error

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
        """
        Updates the weights of the network
        """

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
        """
        Trains the neural network using the provided data

        Parameters
        ----------
        data : list
            Data is a list of data_samples, where each data_sample is a list of input and output.
            For Eg. A typical XOR dataset looks like: 
            XOR_data = [
                [
                    [0, 0],
                    [0]
                ],
                [
                    [0, 1],
                    [1]
                ],
                [
                    [1, 0],
                    [1]
                ],
                [
                    [1, 1],
                    [0]
                ]
            ]
        size : Integer
            Length of the dataset
        
        [MAX_EPOCHS] : Integer
            It is an optional parameter.
            How many iterations should it train for?
        
        graph : bool
            If it is true, an epoch vs error will be displayed after the network is trained.

        Returns
        -------
        Doesn't return anything
        """
        self.epochs_taken = MAX_EPOCHS
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
                out, hidden_out = self.feedForward(inp)
                # Backpropagate the error
                out_errors, total_error = self.backpropagate(target)
                # Update weights
                self.update_weights()

                print(sample, hidden_out, out, out_errors)
                # self.print_weights()
                self.MSE += total_error/self.O
            # self.MSE /= size
            all_erros.append(self.MSE)
                
            print()
        if graph :
            self.epoch_vs_error(all_erros, MAX_EPOCHS)
            
    def predict(self, inputs) :
        """
        Predicts the network's output.

        Parameters
        ----------
        inputs : list
            It is a vector for which the network has to predict the output
        
        Returns
        -------
        output : list
            The predicted output that corresponds to the given input
        """
        output, _hidden_outputs = self.feedForward(inputs)
        return output

    def epoch_vs_error(self, all_erros, epochs) :
        """
        Display epoch vs error graph

        Parameters
        ----------
        all_errors : list
            Training error in each epoch
        epochs : integer
            Number of epochs taken to train the network
        """
        plt.title("Epoch vs Error")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        all_epochs = [i+1 for i in range(epochs)]

        plt.plot(all_epochs, all_erros)
        plt.show()

    def evaluate(self) :
        """
        Shows the network's details.
        """
        print("\n\t=-=-=-=-= Model Evaluation =-=-=-=-=-")
        print("\tModel is trained for", self.epochs_taken, "Epochs")
        print("\tMean Squared Error(MSE): ", self.MSE)
        self.confidence = (1 - self.MSE)*100
        print("\tModel Confidence: ", self.confidence)
        if self.confidence < 75 :
            print("\tModel doesn't seem to fit the data. Try increasing epochs.")
        print("\t-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

    def export_network(self, filename) :
        """
        Serializes the neural network and saves it to a file.

        Parameters
        ----------
        filename : str
            file to save the model

        Returns
        -------
        Doesn't return anything
        """
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
        """
        Desrializes the neural network from a saved model

        Parameters
        ----------
        filename : str
            File containing the saved model

        Returns
        -------
        network : NeuralNetwork
            NeuralNetwork object
        """
        try :
            file_to_read = open(filename, 'rb')
        except Exception as e :
            print("Exception Occurred While opening file: ", filename, ": ", e)
            print("Couldn't Load model")
            return None
        network = pickle.load(file_to_read)
        file_to_read.close()

        return network