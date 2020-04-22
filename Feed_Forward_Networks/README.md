
## Installation  
```bash
$ [sudo] pip install feedforwardnet-shine7
``` 

## Usage

```python3
from Neural_Network import NeuralNetwork

# Create a Neural Network
inputs = 2
output_neurons = 1
hidden_layers = 2
each_hidden_nodes = [2, 3]
network = NeuralNetwork(inputs, hidden_layers, output_neurons, each_hidden_nodes)
```
### Building a dataset
Dataset must be python list of data_samples, where each data_sample is a list of input and target.  
For Eg: Input: [1, 1], Target: [1] => [[1, 1], [1]] is a data sample.

A typical XOR function's dataset looks something like :  
```python
>>> XOR_data = 
[
	[			### ####
		[0, 0], # Input   Data
		[0] # Output	 Sample
	],			### ####
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

>>> size = 4 # Length of the data
```
### Training The network
The library provides a *Train* function which accepts the dataset, dataset size, and two optional parameters MAX\_EPOCHS and Graph.
```python3
def Train(dataset, size, MAX_EPOCHS=10000, graph=False) :
	....
	....
```
For Eg: If you want to train your network for 5000 epochs and display epoch vs error graph after training.
```python3
>>> network.Train(XOR_data, size, MAX_EPOCHS=5000, graph=True)
```

### Debugging
If you want to look at the network's weights at any point of time, the library provides a print\_weights function.
```python
>>> network.print_weights()
```

