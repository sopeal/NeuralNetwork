import random
from scipy.special import expit  # fast logistic function

class NeuralNetwork:

    def __init__(self, size, learning_rate=None, momentum=None):
        if learning_rate:
            self.LEARNING_RATE = learning_rate
        else:
            self.LEARNING_RATE = 0.1

        if momentum:
            self.momentum = momentum
        else:
            self.momentum = 0.0

        self.layers = []
        self.init_layers(size)
        self.init_weights(size)

    def init_layers(self, size):
        self.layers = [NeuronLayer(layer_size) for layer_size in size[1:]]

    def init_weights(self, size):
        for j in range(1, len(size)):
            for h in range(size[j]):
                [self.layers[j-1].neurons[h].weights.append(random.random()) for _ in range(size[j-1])]
                [self.layers[j-1].neurons[h].previous_delta.append(0.0) for _ in range(size[j-1])]

    def feed_forward(self, inputs):
        output = inputs
        for i in range(len(self.layers)):
            output = self.layers[i].feed_forward(output)
        return output

    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)
        self.calculate_layers_de_dz(training_outputs)
        self.update_neurons()

    def calculate_layers_de_dz(self, training_outputs):
        self.layers_de_dz = [0] * len(self.layers)
        self.calculate_output_de_dz(training_outputs)
        self.calculate_hidden_de_dz()

    def calculate_output_de_dz(self, training_outputs):
        output_de_dz = [0] * len(self.layers[-1].neurons)
        for o in range(len(self.layers[-1].neurons)):
            output_de_dz[o] = self.layers[-1].neurons[o].de_dz(training_outputs[o])
        self.layers_de_dz[-1] = output_de_dz

    def calculate_hidden_de_dz(self):
        for i in range(2, len(self.layers) + 1):
            hidden_de_dz = [0] * len(self.layers[-i].neurons)
            for h in range(len(self.layers[-i].neurons)):
                hidden_de_dy = 0
                for o in range(len(self.layers[-i + 1].neurons)):
                    hidden_de_dy += self.layers_de_dz[-i + 1][o] * self.layers[-i + 1].neurons[o].weights[h]
                hidden_de_dz[h] = hidden_de_dy * self.layers[-i].neurons[h].dy_dz()
            self.layers_de_dz[-i] = hidden_de_dz

    def update_neurons(self):
        for i in range(1, len(self.layers) + 1):
            for o in range(len(self.layers[-i].neurons)):
                for weight in range(len(self.layers[-i].neurons[o].weights)):
                    de_dw = self.layers_de_dz[-i][o] * self.layers[-i].neurons[o].dz_dw(weight)
                    self.layers[-i].neurons[o].weights[weight] -= (
                                self.LEARNING_RATE * de_dw + self.layers[-i].neurons[o].part_with_momentum(
                            self.momentum, weight))
                    self.layers[-i].neurons[o].previous_delta[weight] = de_dw
                self.layers[-i].neurons[o].bias -= (
                            self.LEARNING_RATE * self.layers_de_dz[-i][o] + self.layers[-i].neurons[o].momentum_bias(
                        self.momentum))
                self.layers[-i].neurons[o].previous_delta_bias = self.LEARNING_RATE * self.layers_de_dz[-i][o]

    def calculate_total_error(self, training_sets):
        total_error = 0

        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.layers[-1].neurons[o].error(training_outputs[o])
        return total_error

    def calculate_cost(self, target):
        cost = 0
        for o in range(len(self.layers[-1].neurons)):
            cost += self.layers[-1].neurons[o].error(target[o])
        return cost




class NeuronLayer:
    def __init__(self, num_neurons):
        self.bias = random.random()
        self.neurons = []
        self.initialize_neurons(num_neurons)

    def initialize_neurons(self, num_neurons):
        [self.neurons.append(Neuron(self.bias)) for _ in range(num_neurons)]

    def feed_forward(self, inputs):
        return [neuron.calculate_output(inputs) for neuron in self.neurons]


class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []
        self.previous_delta = []
        self.previous_delta_bias = 0.0

    def calculate_output(self, inputs):
        self.inputs = inputs
        total = 0
        for input, weight in zip(self.inputs, self.weights):
            total += input * weight
        self.output = expit(total + self.bias)
        return self.output

    def de_dz(self, target_output):
        return self.de_do(target_output) * self.dy_dz()

    def error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    def de_do(self, target_output):
        return self.output - target_output

    def dy_dz(self):
        return self.output * (1 - self.output)

    def dz_dw(self, index):
        return self.inputs[index]

    def part_with_momentum(self, momentum, which_weight):
        return self.previous_delta[which_weight] * momentum

    def momentum_bias(self, momentum):
        return self.previous_delta_bias * momentum
