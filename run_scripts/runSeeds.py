import pandas as pd

csv = pd.read_csv("seeds_dataset.csv")


data = csv.values.tolist()

import numpy as np
def normalize_data(data_list):
    data = np.asarray(data_list)
    col_maxes = data.max(axis=0)
    return (data / col_maxes[np.newaxis, :]).tolist()

splitted = []
for value in data:
    tmp = value[0].split("\t")
    tmp2 = []
    for str in tmp:
        if str != '':
            tmp2.append(float(str))
    splitted.append(tmp2)
splitted = normalize_data(splitted)

proper_format = []
for value in splitted:
    tmp = [0] * 2
    tmp_target = [0, 0, 0]
    tmp_target[(int(value[7]*3.2)- 1)] = 1
    tmp[0] = value[0:7]
    tmp[1] = tmp_target
    proper_format.append(tmp)

import random

random.shuffle(proper_format)
training_sets = proper_format[0:140].copy()
test_sets = proper_format[140:len(proper_format)].copy()

from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt
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

print("Exp. 4 - various iterations.")
test_case(500, [7,7,3], 0.5, 0.3)
test_case(500, [7,7,3], 0.5, 0.3)
test_case(500, [7,7,3], 0.5, 0.3)
print()
test_case(1500, [7,7,3], 0.5, 0.3)
test_case(1500, [7,7,3], 0.5, 0.3)
test_case(1500, [7,7,3], 0.5, 0.3)
print()
test_case(5000, [7,7,3], 0.5, 0.3)
test_case(5000, [7,7,3], 0.5, 0.3)
test_case(5000, [7,7,3], 0.5, 0.3)


print("Exp. 5 - various learning rate.")
test_case(2000, [7,7,3], 0.2, 0.0)
test_case(2000, [7,7,3], 0.2, 0.0)
test_case(2000, [7,7,3], 0.2, 0.0)
print()
test_case(2000, [7,7,3], 0.5, 0.0)
test_case(2000, [7,7,3], 0.5, 0.0)
test_case(2000, [7,7,3], 0.5, 0.0)
print()
test_case(2000, [7,7,3], 0.8, 0.0)
test_case(2000, [7,7,3], 0.8, 0.0)
test_case(2000, [7,7,3], 0.8, 0.0)
print()


print("Exp. 6 - various momentum.")
test_case(2000, [7,7,3], 0.3, 0.1)
test_case(2000, [7,7,3], 0.3, 0.1)
test_case(2000, [7,7,3], 0.3, 0.1)
print()
test_case(2000, [7,7,3], 0.3, 0.5)
test_case(2000, [7,7,3], 0.3, 0.5)
test_case(2000, [7,7,3], 0.3, 0.5)
print()
test_case(2000, [7,7,3], 0.3, 0.9)
test_case(2000, [7,7,3], 0.3, 0.9)
test_case(2000, [7,7,3], 0.3, 0.9)
print()


print('b')