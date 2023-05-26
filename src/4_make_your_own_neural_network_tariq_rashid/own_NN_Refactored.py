import unittest
import imageService as imgSrv
import numpy
import numpy as np
from typing import List
from collections import deque


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

    def __init__(self, features_x: float = None):
        self.features_x = features_x


class Neuron:
    """
    The hidden layers and the output layer Neurons are where the weights and biases come into play.
    Each neuron in these layers has associated weights that determine the strength and influence
    of its connections with the neurons in the previous layer.
    The weights are used to scale the input values and adjust the importance of each neuron's contribution to the final
    output.
    """

    def __init__(self, weights: list, features_x=None, bias: float = 0.1):
        if features_x is None:
            features_x = []

        self.weights = weights
        self.features_x = features_x
        self.bias = bias

    def update_weights(self, weights: list):
        self.weights = weights


class NeuralNetwork:

    # initialise the neural network
    def __init__(self):
        self.layers = list()

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
            hidden_layers_neurons = list()
            for n in range(0, nodes):
                # initialize weights with gaussian distribution
                weights_gauss = self.link_weight_ih_and_ho_gaussian_distribution(prev_layer_neurons, nodes)
                hidden_layers_neurons.append(Neuron(weights_gauss))

            self.layers.append(hidden_layers_neurons)

    @staticmethod
    def activation_function_sigmoid(x):
        """
        Sigmoid activation function.
        :param x:
        :return:
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def link_weight_ih_and_ho_gaussian_distribution(hidden_nodes: int, input_nodes: int):
        """"
        They sample the weights from a normal probability distribution centered around zero
        and with a standard deviation that is related to the number of incoming links into a node
        1/√(number of incoming links).
        In other words weights 'w' will be initialized with 67% probability in normal distribution window

        See https://www.techtarget.com/whatis/definition/normal-distribution
        """
        return numpy.random.normal(0.0, pow(hidden_nodes, -0.5), (hidden_nodes, input_nodes))

    # TODO old code

    def link_weight_ih_and_ho_uniform_distribution(self):
        """
        Link weight matrices, weight 'w' for input and hidden, and 'w' for hidden and output
        weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        w11 -> w21
        w12 -> w22
        etc ...
        For simplicity, we’ll simply subtract 0.5 for range between − 0.5 to +0.5

        See https://www.investopedia.com/terms/u/uniform-distribution.asp

        :return:
        """

        self.w_input_hidden = (numpy.random.rand(self.hidden_nodes, self.input_nodes) - 0.5)
        self.w_hidden_output = (numpy.random.rand(self.output_nodes, self.hidden_nodes) - 0.5)

    def link_weight_ih_and_ho_gaussian_distribution(self):
        """"
        They sample the weights from a normal probability distribution centered around zero
        and with a standard deviation that is related to the number of incoming links into a node
        1/√(number of incoming links).
        In other words weights 'w' will be initialized with 67% probability in normal distribution window

        See https://www.techtarget.com/whatis/definition/normal-distribution
        """

        self.w_input_hidden = numpy.random.normal(0.0,
                                                  pow(self.hidden_nodes, -0.5),
                                                  (self.hidden_nodes, self.input_nodes))

        self.w_hidden_output = numpy.random.normal(0.0,
                                                   pow(self.output_nodes, -0.5),
                                                   (self.output_nodes, self.hidden_nodes))

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
        hidden_outputs = self.activation_function_sigmoid(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.w_hidden_output, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function_sigmoid(final_inputs)

        # output layer error is the (target - actual)
        # TODO play around additional loss functions (MSE, etc)
        output_errors = targets - final_outputs

        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        # TODO errors negative sign error
        hidden_errors = numpy.dot(self.w_hidden_output.T, output_errors)

        # TODO do not repeat yourself, use single function - "backpropagation_algorithm"
        # update the weights for the links between the hidden and output layers
        self.w_hidden_output += self.learning_grate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                                                numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.w_input_hidden += self.learning_grate * numpy.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass

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
        hidden_outputs = self.activation_function_sigmoid(hidden_inputs)
        # TODO query_layer "do not repit yourself"
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.w_hidden_output, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function_sigmoid(final_inputs)

        return final_outputs


class NeuralNetworkTest(unittest.TestCase):
    def setUp(self):
        # Create an instance of the Calculator class for each test
        self.nn = NeuralNetwork()

    def test_add_input_layer(self):
        features = [1, 2, 3]
        self.nn.add_input_layer(features)
        self.assertEqual(len(self.nn.layers), 1)
