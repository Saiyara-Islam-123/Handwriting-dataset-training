import matplotlib.pyplot as plt
import random
import torch
import create_dataset

def sampled_avg_distance(pair, X, y):

    X_first = []
    X_second = []

    for i in range(y.size(0)):
        row = y[i]
        if row in pair:
            if row == pair[0]:
                X_first.append(X[i])
            if row == pair[1]:
                X_second.append(X[i])

    distances = []
    for i in range(4000):
        index1 = random.randint(0, 400)
        index2 = random.randint(0, 400)



        distances.append(abs(torch.norm(X_first[index1]- X_second[index2])))

    return (sum(distances) / len(distances)).item()

#I basically find the Euclidean distance between two random datapoints from these two bigger matrices.



if __name__ == '__main__':
    print(sampled_avg_distance((4,9),create_dataset.get_inputs(), create_dataset.get_labels()))
    print(sampled_avg_distance((4, 4), create_dataset.get_inputs(), create_dataset.get_labels()))