import random
from scipy.special import expit # fast logistic function


class NeuralNetwork:

    def __init__(self, size):
        self.LEARNING_RATE = 0.5
        self.layers = []
        self.init_layers(size)
        self.init_weights(size)

    def init_layers(self, size):
        for i in range(1, len(size)):
            self.layers.append(NeuronLayer(size[i]))

    def init_weights(self, size):
        for j in range(1, len(size)):
            for h in range(size[j]):
                for i in range(size[j-1]):
                    self.layers[j-1].neurons[h].weights.append(random.random())

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
                    self.layers[-i].neurons[o].weights[weight] -= self.LEARNING_RATE * de_dw
                self.layers[-i].neurons[o].bias -= self.LEARNING_RATE * self.layers_de_dz[-i][o]

    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.layers[-1].neurons[o].error(training_outputs[o])
        return total_error


class NeuronLayer:
    def __init__(self, num_neurons):
        self.bias = random.random()
        self.initialize_neurons(num_neurons)

    def initialize_neurons(self, num_neurons):
        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs


class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def calculate_output(self, inputs):
        self.inputs = inputs
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
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
