#I'll add noise!
import numpy as np
import torch

import create_dataset
import random

random.seed(99)

l = [
create_dataset.get_inputs()[88].reshape(28, 28), #0
create_dataset.get_inputs()[24].reshape(28, 28),
create_dataset.get_inputs()[199].reshape(28, 28),
create_dataset.get_inputs()[44].reshape(28, 28),
create_dataset.get_inputs()[2].reshape(28, 28),
create_dataset.get_inputs()[0].reshape(28, 28),
create_dataset.get_inputs()[90].reshape(28,28),
create_dataset.get_inputs()[15].reshape(28,28),
create_dataset.get_inputs()[55].reshape(28, 28),
create_dataset.get_inputs()[33].reshape(28, 28), #9

]

def synthetic_data(m):
    m = m.tolist()

    for i in range(len(m)):
        for j in range(len(m[i])):
            m[i][j] += np.random.normal(0, 0.3)


    return torch.tensor(m)

def generate(m, label, num_variations):
    synth_dataset = []
    labels = []

    for i in range(num_variations):
        synth_matrix = synthetic_data(m)
        synth_dataset.append(synth_matrix)
        labels.append(label)

    return torch.stack(synth_dataset), torch.tensor(labels)

def shuffle ():
    Xgen = []
    ygen = []
    i =0
    for mat in l:
        mats, labels = generate(mat, torch.tensor(i), 6000)
        Xgen.append(mats)
        ygen.append(labels)
        i += 1

    random.shuffle(Xgen)
    random.shuffle(ygen)

    return torch.cat(Xgen, dim=0), torch.cat(ygen, dim=0)


if __name__ == '__main__':
    #label_9 = create_dataset.get_labels()[33]
    #label_4 = create_dataset.get_labels()[2]

    #y = create_dataset.get_labels()


    print(shuffle()[0].shape)
    print(shuffle()[1].shape)



    #var_9 = generate(mat_9, label_9, 10)
    #var_4 = generate(mat_4, label_4,10)

    #print(var_9)
    #print(var_4)