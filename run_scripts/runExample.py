import matplotlib.pyplot as plt
import random
from NeuralNetwork import NeuralNetwork

training_sets = [
    [[1, 0, 0, 0], [1, 0, 0, 0]],
    [[0, 1, 0, 0], [0, 1, 0, 0]],
    [[0, 0, 1, 0], [0, 0, 1, 0]],
    [[0, 0, 0, 1], [0, 0, 0, 1]]
]

def test_case(max_error, learning_rate, momentum):
    nn = NeuralNetwork([4, 2, 4], learning_rate, momentum)
    error = 5
    history_errors = []
    while error >= max_error:
        training_inputs, training_outputs = random.choice(training_sets)
        nn.train(training_inputs, training_outputs)
        error = nn.calculate_total_error(training_sets)
        history_errors.append(error)
    print("Max error: ", max_error, " Learning rate: ", learning_rate, " Momentum: ", momentum," Number of iterations: ", len(history_errors))
    plt.plot(history_errors)
    plt.xlabel("Number of iterations.")
    plt.ylabel("Total error.")
    plt.title("History of total errors.")
    plt.show()


test_case(0.03, 0.9, 0.0)
test_case(0.03, 0.9, 0.6)
test_case(0.03, 0.6, 0.0)
test_case(0.03, 0.6, 0.9)
test_case(0.03, 0.2, 0.0)
test_case(0.03, 0.2, 0.9)
