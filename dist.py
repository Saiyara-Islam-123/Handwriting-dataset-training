import random

import pandas as pd
import torch.nn.functional
import create_dataset

random.seed(0)

def avg_distance(pair, X, y):

    X_first = []
    X_second = []
    print(X.size(), y.size())

    for i in range(100):
        row = y[i]
        if row in pair:
            if row == pair[0]:
                X_first.append(X[i])
            if row == pair[1]:
                X_second.append(X[i])


    distances = []

    cat_1, cat_2 = pair

    for i in range(len(X_first)):
        for j in range(len(X_second)):

            if cat_1 == cat_2 and i == j: #for within dist calculation skip calculating distnace with self
                continue

            mat_1 = X_first[i]
            mat_2 = X_second[j]

            mat_1_flattened = mat_1.view(mat_1.size(0), -1)
            mat_2_flattened = mat_2.view(mat_2.size(0), -1)

            mat_1_flattened_normalized = torch.nn.functional.normalize(mat_1_flattened, p=2, dim=1)
            mat_2_flattened_normalized = torch.nn.functional.normalize(mat_2_flattened, p=2, dim=1)

            distances.append(torch.norm(mat_1_flattened_normalized - mat_2_flattened_normalized).detach().numpy())


    return sum(distances)/len(distances)




#I basically find the Euclidean distance between two random datapoints from these two bigger matrices.



if __name__ == '__main__':
    between = avg_distance((1,0),create_dataset.get_inputs(), create_dataset.get_labels())
    within_1 =  avg_distance((1,1),create_dataset.get_inputs(), create_dataset.get_labels())
    within_0 = avg_distance((0, 0), create_dataset.get_inputs(), create_dataset.get_labels())
    #print(sampled_avg_distance((4, 4), create_dataset.get_inputs(), create_dataset.get_labels()))
    df = pd.DataFrame()
    df["between"] = [between]
    df["within_1"] = [within_1]
    df["within_0"] = [within_0]

    df.to_csv("no_train_0_1.csv", index=False)