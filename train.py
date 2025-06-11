import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

import neural_networks
from torch import optim
import torch.nn as nn
from dist import *
from accuracy import *
import seaborn as sb

import matplotlib.pyplot as plt

def scale(x):
    return 100 * x

def train_unsup(tup, batch_size, lr):
    f, s = tup

    pair_avg_distances = {}
    pair_avg_distances[(f, f)] = []
    pair_avg_distances[(f, s)] = []
    pair_avg_distances[(s, s)] = []

    model = neural_networks.AutoEncoder()
    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    inputs = create_dataset.get_inputs()

    batches = list(torch.split(inputs, batch_size))
    batches_of_labels = list(torch.split(create_dataset.get_labels(), batch_size))

    print("\nUnsupervised part!")

    for epoch in range(1):

        for i in range(len(batches)):
            optimizer.zero_grad()

            outputs = model(batches[i])
            encoder_output = model.encoded
            labels = batches_of_labels[i]

            loss = loss_fn(outputs, batches[i])



            pair_avg_distances[(f, f)].append(avg_distance((f, f), encoder_output, labels))
            pair_avg_distances[(f, s)].append(avg_distance((f, s), encoder_output, labels))
            pair_avg_distances[(s, s)].append(avg_distance((s, s), encoder_output, labels))

            torch.save(model.state_dict(),
                       "Batch: " + str(i) + " " + str(batch_size) + " " + str(lr) + " unsup_model.pt")


            print(loss.item())

            loss.backward()
            optimizer.step()

    df = pd.DataFrame()
    df["within " + str(f)] = pair_avg_distances[(f, f)]
    df["within " + str (s)] = pair_avg_distances[(s, s)]
    df["between"] = pair_avg_distances[(f, s)]

    df.to_csv("Unsup " + str(f) + " " + str(s) + " lr=" + str(lr)  , index=False)

def train_sup(unsup_model, tup, lr, batch_size):
    f,s = tup
    pair_avg_distances = {}
    pair_avg_distances[(f, f)] = []
    pair_avg_distances[(f, s)] = []
    pair_avg_distances[(s, s)] = []

    accs= []

    model_2 = neural_networks.LastLayer(unsup_model)

    model_2.train()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_2.parameters(), lr=lr, weight_decay=0.0001)

    inputs = create_dataset.get_inputs()

    batches = list(torch.split(inputs, batch_size))
    batches_of_labels = list(torch.split(create_dataset.get_labels(), batch_size))

    print("\nSupervised part!")

    for epoch in range(1):

        for i in range(len(batches)):
            optimizer.zero_grad()

            outputs = model_2(batches[i])
            encoder_outputs = model_2.autoencoder_output

            labels = batches_of_labels[i]

            loss = loss_fn(outputs, batches_of_labels[i])


            pair_avg_distances[(f, f)].append(avg_distance((f, f), encoder_outputs, labels))
            pair_avg_distances[(f, s)].append(avg_distance((f, s), encoder_outputs, labels))
            pair_avg_distances[(s,s)].append(avg_distance((s, s), encoder_outputs, labels))

            torch.save(model_2.state_dict(), "sup_weights/Batch: " + str(i) + " " + str(batch_size) + " " + str(lr) + " sup_model.pt")


            loss.backward()
            accuracy = acc(model_2)
            accs.append(accuracy)
            optimizer.step()

    df = pd.DataFrame()
    df["within " + str(f)] = pair_avg_distances[(f, f)]
    df["within " + str(s)] = pair_avg_distances[(s, s)]
    df["between"] = pair_avg_distances[(f, s)]
    df["Accuracy"] = accs

    df.to_csv("Sup Slowed down " + str(f) + " " + str(s) + " lr=" + str(lr), index=False)





def plot(tup, num_unsup_rows, num_sup_rows):
    f,s = tup
    df_unsup = pd.read_csv("Unsup1 0.csv")
    df_unsup = df_unsup.tail(num_unsup_rows)


    df_sup = pd.read_csv("Sup Slowed down1 0 lr=0.001")
    df_sup = df_sup.head(num_sup_rows)


    accs = df_sup["Accuracy"].apply(scale)
    df_sup.drop(columns=["Accuracy"], inplace=True)

    df_whole = pd.concat([df_unsup, df_sup])
    x = []
    for i in range(num_unsup_rows+num_sup_rows):
        x.append(i)
    x2 = []
    for i in range(num_unsup_rows, num_unsup_rows+num_sup_rows):
        x2.append(i)

    fig, ax1 = plt.subplots()

    ax1.plot(x, df_whole['within ' + str(f)], color="limegreen", label='within ' + str(f), alpha = 0.8)
    ax1.plot(x, df_whole['within ' + str(s)], color="green", label='within ' + str(s), alpha=0.8)
    ax1.plot(x, df_whole['between'], color="blue", label="between", alpha=0.8)


    ax1.axvline(x=num_unsup_rows-1, color='r', linestyle='--')
    ax1.set_ylabel('Distance')
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(x2, accs, color="coral", label="accuracy")
    ax2.set_ylabel('Accuracy')


    plt.xlabel("Epoch")
    plt.title(str(f) + " " + str(s) + " distances and accuracy across batches")
    plt.savefig("Acc distance plot 0 1.png")
    plt.show()



if __name__ == '__main__':
    batch_size = 100
    lr = 0.0001

    train_unsup(tup=(1,0), batch_size=batch_size, lr=lr)
    #unsup_model = neural_networks.AutoEncoder()
    #unsup_model.load_state_dict(torch.load(str(batch_size) + " " + str(lr) + " unsup_model.pt"))
    #train_sup(unsup_model, tup=(1,0), batch_size=batch_size, lr=lr)

    #plot(tup=(1,0), num_unsup_rows=30, num_sup_rows=30)