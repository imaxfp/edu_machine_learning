import unittest
from typing import List
from collections import deque
import numpy as np


# You own Neural Network implementation
class NeuronInputLayer:
    """
    In a typical neural network architecture, the input layer neurons do not have weights associated with them.
    The input layer is responsible for receiving the raw input data, such as images, text, or numerical values,
    and transmitting that information to the subsequent layers of the network.
    The input layer simply passes the input data directly to the next layer, which is usually a hidden layer.

    However "recurrent neural network" (RNN), specifically a "neural network with input weights" or "weighted input RNN."
    In a weighted input RNN, the input layer neurons can have individual weights associated with them.
    These weights determine the importance or relevance of each input feature to the network's behavior.
    This allows the network to assign different levels of significance to different input dimensions.
    """

    def __init__(self, feature_x: float = None):
        self.feature_x = feature_x


class Neuron:
    """
    The hidden layers and the output layer Neurons are where the weights and biases come into play.
    Each neuron in these layers has associated weights that determine the strength and influence
    of its connections with the neurons in the previous layer.
    The weights are used to scale the input values and adjust the importance of each neuron's contribution to the final
    output.
    """

    def __init__(self, weights: list, features_x: list = None, bias: float = 0.1):
        self.weights = weights
        self.features_x = features_x
        self.bias = bias

    def update_weights(self, weights: list):
        self.weights = weights

    def update_features(self, x: list):
        self.features_x = x


class NeuralNetwork:
    # initialise the neural network
    def __init__(self):
        self.layers = []

    def add_input_layer(self, features: List):
        if self.layers.__len__() == 0:
            input_layer_neurons = [NeuronInputLayer(f) for f in features]
            self.layers.append(input_layer_neurons)

    def add_layer(self, nodes: int):
        # check if we have input layer
        if self.layers.__len__() > 0:
            # get amount of neurons of the previous layer
            layers_count = len(self.layers)
            prev_layer_neurons = len(self.layers[layers_count - 1])
            # create new layer based on previous layer
            hidden_layers_neurons = []
            for _ in range(0, nodes):
                # initialize weights with gaussian distribution
                weights_gauss = self.link_weight_ih_and_ho_gaussian_distribution(
                    prev_layer_neurons, nodes
                )
                hidden_layers_neurons.append(Neuron(weights_gauss))

            self.layers.append(hidden_layers_neurons)

    @staticmethod
    def activation_fun_sig(x):
        """
        Sigmoid activation function.
        :param x:
        :return:
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def link_weight_ih_and_ho_gaussian_distribution(hidden_nodes: int, input_nodes: int):
        """_summary_
                They sample the weights from a normal probability distribution centered around zero
        and with a standard deviation that is related to the number of incoming links into a node
        1/√(number of incoming links).
        In other words weights 'w' will be initialized with 67% probability in normal distribution window

        See https://www.techtarget.com/whatis/definition/normal-distribution

        Args:
            hidden_nodes (int): _description_
            input_nodes (int): _description_

        Returns:
            _type_: _description_
        """
        return np.random.normal(
            0.0, pow(hidden_nodes, -0.5), (hidden_nodes, input_nodes)
        )

    def forward_propagation(self):
        """_summary_
        Implementation of forward propagation algorithm

        Args:
            inputs_list (_type_): _description_
            targets_list (_type_): _description_
        """
        input_layer_features = [x.feature_x for x in self.layers[0]]
        layer_output = []
        for layer in self.layers[1:]:
            for neurone in layer:
                neurone.features_x = input_layer_features
                n_weights = np.array(neurone.weights, ndmin=2).T
                n_features = np.array(neurone.features_x, ndmin=2).T
                # Matrices multiplication see http
                neuron_output = self.activation_fun_sig(np.dot(n_features, n_weights))
                layer_output.append(neuron_output)
            input_layer_features = layer_output.copy()
            layer_output.clear()    
                

    def train(self, inputs_list, targets_list):
        """
        Train the neural network
        :param inputs_list:
        :param targets_list:
        :return:
        """

        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # TODO do not repeat yourself
        # calculate signals into hidden layer. If both `a` and `b` are 2-D arrays, it is matrix multiplication
        hidden_inputs = numpy.dot(self.w_input_hidden, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_fun_sig(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.w_hidden_output, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_fun_sig(final_inputs)

        # output layer error is the (target - actual)
        # TODO play around additional loss functions (MSE, etc)
        output_errors = targets - final_outputs

        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        # TODO errors negative sign error
        hidden_errors = numpy.dot(self.w_hidden_output.T, output_errors)

        # TODO do not repeat yourself, use single function - "backpropagation_algorithm"
        # update the weights for the links between the hidden and output layers
        self.w_hidden_output += self.learning_grate * numpy.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            numpy.transpose(hidden_outputs),
        )

        # update the weights for the links between the input and hidden layers
        self.w_input_hidden += self.learning_grate * numpy.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            numpy.transpose(inputs),
        )

    def query(self, inputs_list):
        """
        The query() function takes the input to a neural network and returns the network’s output.
        Pass the input signals from the input layer of nodes, through the hidden layer and out of the final output layer
        :return:
        """

        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.w_input_hidden, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_fun_sig(hidden_inputs)
        # TODO query_layer "do not repit yourself"
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.w_hidden_output, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_fun_sig(final_inputs)

        return final_outputs
    
    def print_nn(self):
        
        pass
        


# TESTS


class NeuralNetworkBasicTest(unittest.TestCase):
    def setUp(self):
        self.nn = NeuralNetwork()

    def test_add_input_layer(self):
        self.nn.add_input_layer(features=[1, 21, 31])
        self.assertEqual(len(self.nn.layers[0]), 3)

    def test_add_layer(self):
        self.nn.add_input_layer(features=[1, 0, 3])
        self.nn.add_layer(nodes=10)
        # check amount of neurons for the layer 1
        self.assertEqual(len(self.nn.layers[1]), 10)
        # check amount of weights for the neuron of layer 1. Previous layer - 0 layer, has 3 feature.
        neuron_first_layer = self.nn.layers[1][0]
        self.assertEqual(len(neuron_first_layer.weights), 3)

    def test_forward_propagation(self):
        self.nn.add_input_layer(features=[1, 2, 3])
        self.nn.add_layer(3)
        self.nn.forward_propagation()


# Run the tests
if __name__ == "__main__":
    unittest.main()
