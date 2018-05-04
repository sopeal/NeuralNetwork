import matplotlib.pyplot as plt
import random
from NeuralNetwork import NeuralNetwork
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
import numpy as np

def normalize_data(data):
    col_maxes = data.max(axis=0)
    return data / col_maxes[np.newaxis, :]

iris = load_iris()
iris = shuffle(iris)
data = normalize_data(iris.data)
data = data.tolist()
target = iris.target.tolist()

training_sets = []
for i in range(100):
    tmp = [0, 0]
    tmp_target = [0, 0, 0]
    tmp_target[target[i]] = 1
    tmp[0] = data[i]
    tmp[1] = tmp_target
    training_sets.append(tmp)

test_sets = []
for i in range(100, len(data)):
    tmp = [0, 0]
    tmp_target = [0, 0, 0]
    tmp_target[target[i]] = 1
    tmp[0] = data[i]
    tmp[1] = tmp_target
    test_sets.append(tmp)

def test_case(iterations, size, learning_rate, momentum):
    nn = NeuralNetwork(size, learning_rate, momentum)
    history_errors = []
    i = 0
    while i < iterations :
        training_inputs, training_outputs = random.choice(training_sets)
        nn.train(training_inputs, training_outputs)
        if i % (iterations / 100) == 0:
            error = nn.calculate_total_error(training_sets)
            history_errors.append(error)
        i = i + 1

    accurate_outputs = 0.0
    for i in range(len(test_sets)):
        output = nn.feed_forward(test_sets[i][0])
        max_index_output = output.index(max(output))
        max_index_target = test_sets[i][1].index(max(test_sets[i][1]))
        if max_index_output == max_index_target:
            accurate_outputs = accurate_outputs + 1.0
    test_rate = accurate_outputs / float(len(test_sets))

    print(nn.feed_forward(training_sets[0][0]))
    print(" Learning rate: ", learning_rate, " Momentum: ", momentum," Number of iterations: ", len(history_errors), "Test rate ", test_rate)
    plt.plot(history_errors)
    plt.xlabel("Number of iterations.")
    plt.ylabel("Total error.")
    plt.title("History of total errors.")
    plt.show()

test_case(10000, [4,2,3], 0.5, 0.0)
test_case(10000, [4,7,3], 0.5, 0.0)
test_case(10000, [4,12,3], 0.5, 0.0)
test_case(10000, [4,20,3], 0.5, 0.0)

