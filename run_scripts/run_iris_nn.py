import matplotlib.pyplot as plt
import random
from NeuralNetwork import NeuralNetwork
from sklearn.datasets import load_iris
import numpy as np

def normalize_data(data):
    col_maxes = data.max(axis=0)
    return data / col_maxes[np.newaxis, :]

iris = load_iris()
data = normalize_data(iris.data)
data = data.tolist()
target = iris.target.tolist()

proper_format = []
for i in range(len(data)):
    tmp = [0, 0]
    tmp_target = [0, 0, 0]
    tmp_target[target[i]] = 1
    tmp[0] = data[i]
    tmp[1] = tmp_target
    proper_format.append(tmp)

random.shuffle(proper_format)

training_sets = proper_format[0:100].copy()
test_sets = proper_format[100:len(proper_format)].copy()

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

    print("iterations = ", iterations, "size = ", size, "Learning rate: ", learning_rate, " Momentum: ", momentum,"Test rate ", test_rate)
    plt.plot(history_errors)
    plt.xlabel("History points.")
    plt.ylabel("Total error.")
    plt.title("History of total errors.")
    plt.show()

print("Exp. 1 - various size.")
test_case(1000, [4,2,3], 0.5, 0.0)
test_case(1000, [4,2,3], 0.5, 0.0)
test_case(1000, [4,2,3], 0.5, 0.0)
print()
test_case(1000, [4,7,3], 0.5, 0.0)
test_case(1000, [4,7,3], 0.5, 0.0)
test_case(1000, [4,7,3], 0.5, 0.0)
print()
test_case(1000, [4,12,3], 0.5, 0.0)
test_case(1000, [4,12,3], 0.5, 0.0)
test_case(1000, [4,12,3], 0.5, 0.0)
print()
test_case(1000, [4,20,3], 0.5, 0.0)
test_case(1000, [4,20,3], 0.5, 0.0)
test_case(1000, [4,20,3], 0.5, 0.0)


print("Exp. 2 - various learning rate.")
test_case(1000, [4,2,3], 0.2, 0.0)
test_case(1000, [4,2,3], 0.2, 0.0)
test_case(1000, [4,2,3], 0.2, 0.0)
print()
test_case(1000, [4,2,3], 0.4, 0.0)
test_case(1000, [4,2,3], 0.4, 0.0)
test_case(1000, [4,2,3], 0.4, 0.0)
print()
test_case(1000, [4,2,3], 0.6, 0.0)
test_case(1000, [4,2,3], 0.6, 0.0)
test_case(1000, [4,2,3], 0.6, 0.0)
print()
test_case(1000, [4,2,3], 0.8, 0.0)
test_case(1000, [4,2,3], 0.8, 0.0)
test_case(1000, [4,2,3], 0.8, 0.0)

print("Exp. 3 - various momentum.")
test_case(1000, [4,2,3], 0.2, 0.3)
test_case(1000, [4,2,3], 0.2, 0.3)
test_case(1000, [4,2,3], 0.2, 0.3)
print()
test_case(1000, [4,2,3], 0.2, 0.5)
test_case(1000, [4,2,3], 0.2, 0.5)
test_case(1000, [4,2,3], 0.2, 0.5)
print()
test_case(1000, [4,2,3], 0.2, 0.7)
test_case(1000, [4,2,3], 0.2, 0.7)
test_case(1000, [4,2,3], 0.2, 0.7)
print()
test_case(1000, [4,2,3], 0.2, 0.9)
test_case(1000, [4,2,3], 0.2, 0.9)
test_case(1000, [4,2,3], 0.2, 0.9)

