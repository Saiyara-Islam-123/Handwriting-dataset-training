#I'll add noise!
import numpy as np
import torch

import matplotlib.pyplot as plt
import create_dataset

def synthetic_data(m):
    m = m.tolist()

    for i in range(len(m)):
        for j in range(len(m[i])):
            m[i][j] += np.random.normal(0, 0.3)


    return torch.tensor(m)

def generate(m, label, num_variations):
    synth_dataset = []

    for i in range(num_variations):
        synth_matrix = synthetic_data(m)
        synth_dataset.append(synth_matrix)

    return torch.stack(synth_dataset)

if __name__ == '__main__':
    mat_9 = create_dataset.get_inputs()[33].reshape(28, 28)
    mat_4 = create_dataset.get_inputs()[2].reshape(28, 28)

    var_9 = generate(mat_9, 10000)
    var_4 = generate(mat_4, 10000)