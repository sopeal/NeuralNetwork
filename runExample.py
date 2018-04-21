import matplotlib.pyplot as plt
import random
from NeuralNetwork import NeuralNetwork

training_sets = [
    [[1, 0, 0, 0], [1, 0, 0, 0]],
    [[0, 1, 0, 0], [0, 1, 0, 0]],
    [[0, 0, 1, 0], [0, 0, 1, 0]],
    [[0, 0, 0, 1], [0, 0, 0, 1]]
]

nn = NeuralNetwork([4, 2, 4])
iterations = 10000
history_errors = []
for i in range(iterations):
    training_inputs, training_outputs = random.choice(training_sets)
    nn.train(training_inputs, training_outputs)
    history_errors.append(nn.calculate_total_error(training_sets))


plt.plot(history_errors)
plt.show()

print(nn.feed_forward(training_sets[0][0]))
print(nn.feed_forward(training_sets[1][0]))
print(nn.feed_forward(training_sets[2][0]))
print(nn.feed_forward(training_sets[3][0]))