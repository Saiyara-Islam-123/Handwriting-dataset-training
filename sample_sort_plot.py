import torch
import random
from sklearn.metrics.pairwise import cosine_similarity
from functools import cmp_to_key
from plotting_X import filter
from create_dataset import *
import pandas as pd
import json
import ast
import matplotlib.pyplot as plt
import re

random.seed(920)


def sample(X_filtered1, X_filtered_2, num_samples):
    total_datapoints1 = X_filtered1.shape[0]
    total_datapoints2 = X_filtered_2.shape[0]
    pairs = {}

    for i in range(num_samples):
        rolled_num1 = random.randint(0, total_datapoints1 - 1)
        rolled_num2 = random.randint(0, total_datapoints2 - 1)
        while rolled_num1 == rolled_num2:
            rolled_num2 = random.randint(0, total_datapoints2 - 1)

        if (rolled_num1, rolled_num2) not in pairs and (rolled_num1, rolled_num2) not in pairs :
            pairs[(rolled_num1, rolled_num2)] = (1 - cosine_similarity(X_filtered1[rolled_num1].reshape(1, 784),
                                                                      X_filtered_2[rolled_num2].reshape(1, 784))).tolist()[0]
    return pairs

def sort(pairs): #for one category
    #X and Y are tensors
    #######Sorting pairs

    def compare(t1, t2):
        if pairs[t1] < pairs[t2]:
            return -1
        elif pairs[t1] > pairs[t2]:
            return 1
        else:
            return 0
    print(pairs)
    sorted_pairs_list = sorted(pairs, key=cmp_to_key(compare), reverse=True)


    sorted_dict = {}
    for j in range(len(sorted_pairs_list)):

        sorted_dict[(sorted_pairs_list[j])] = pairs[sorted_pairs_list[j]]


    return sorted_dict

def pairs_to_cosine_similarity(pairs_list, Xf1, Xf2):
    pairs_dict = {}

    for p in pairs_list:
        index_1 = p[0]
        index_2 = p[1]

        pairs_dict[p] = (1 - cosine_similarity(Xf1[index_1].reshape(1, 128),
                               Xf2[index_2].reshape(1, 128))).tolist()[0]

    return pairs_dict


def format(x):
    pattern = r"[\[\]]"
    x = re.sub(pattern, '', x)
    return float(x)

if __name__ == "__main__":

    df_pre = pd.read_csv("no_training_distance 4.csv").iloc[:10000]
    df_unsup = pd.read_csv("unsup_training_distance 4.csv").iloc[:10000]
    df_sup = pd.read_csv("sup_training_distance 4.csv").iloc[:10000]

    df_pre["1-cos sim"] = df_pre['1-cos sim'].map(format)
    df_unsup["1-cos sim"] = df_unsup['1-cos sim'].map(format)
    df_sup["1-cos sim"] = df_sup['1-cos sim'].map(format)


    fig, ax = plt.subplots()
    ax.plot(df_pre["Pair"], df_pre["1-cos sim"], label="No Train", color="darkgreen")
    ax.plot(df_unsup["Pair"], df_unsup["1-cos sim"], label="Unsup", color="lime")
    ax.plot(df_sup["Pair"], df_sup["1-cos sim"], label="Sup", color="darkgreen", alpha=0.3)

    ax.set_xticks([])
    ax.set_ylabel("1-cos similarity")
    ax.set_xlabel("Pairs")
    ax.set_title("Within 4 dist")
    ax.legend()
    plt.show()




    #with open("sup.json", "r") as f:
        #loaded_dict = json.load(f)

    #encoder_output_sup = torch.tensor(loaded_dict["sup outputs"])


    #X = encoder_output_sup
    #y = get_labels()
    #X_filtered1, _ = filter(X, y, [4])
    #X_filtered2, _ = filter(X, y, [4])
    #pairs = sample(X_filtered1, X_filtered2, 10000)
    #sorted_pairs = sort(pairs)
    #df = pd.DataFrame(list(sorted_pairs.items()), columns=["Pair", "1-cos sim"])
    #df.to_csv("no_training_distance 4.csv", index=False)


    #df = pd.read_csv("no_training_distance 4.csv")
    #pairs_list = list(map(ast.literal_eval, df["Pair"].tolist()))

    #pairs = pairs_to_cosine_similarity(pairs_list, X_filtered1, X_filtered2)

    #df = pd.DataFrame(list(pairs.items()), columns=["Pair", "1-cos sim"])
    #df.to_csv("sup_training_distance 4.csv", index=False)

    
    #pairs = sample(X_filtered1, X_filtered2, 10000)
    #sorted_dict = sort(pairs)
    #print(sorted_dict)

    #df = pd.DataFrame(list(sorted_dict.items()), columns=["Pair", "1-cos sim"])
    #df.to_csv("no_training_distance.csv", index=False)
    
