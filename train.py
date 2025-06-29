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

            torch.save(model.state_dict(), "unsup_weights/" + str(i) + " " + str(batch_size) + " " + str(lr) + " unsup_model.pth")


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

            torch.save(model_2.state_dict(), f"sup_weights/{lr}/" +str(i) + " " + str(batch_size) + " " + str(lr) + " sup_model.pth")


            loss.backward()
            accuracy = acc(model_2)
            accs.append(accuracy)
            optimizer.step()

    df = pd.DataFrame()
    df["within " + str(f)] = pair_avg_distances[(f, f)]
    df["within " + str(s)] = pair_avg_distances[(s, s)]
    df["between"] = pair_avg_distances[(f, s)]
    df["Accuracy"] = accs

    df.to_csv("Sup " + str(f) + " " + str(s) + " lr=" + str(lr), index=False)


def plot_per_batch(tup, num_unsup_rows, num_sup_rows, time_step, sup_df, loc):
    f,s = tup
    df_no_train = pd.read_csv("no_train_0_1.csv")


    df_unsup = pd.read_csv("Unsup 1 0 lr=0.05")
    df_unsup = df_unsup.tail(num_unsup_rows)


    df_sup = pd.read_csv(sup_df)
    df_sup = df_sup.head(num_sup_rows)


    accs = df_sup["Accuracy"].apply(scale)
    df_sup.drop(columns=["Accuracy"], inplace=True)

    df_whole = pd.concat([df_no_train, df_unsup, df_sup])
    x = []
    for i in range(1+num_unsup_rows+num_sup_rows):
        x.append(i)
    x2 = []
    for i in range(num_unsup_rows+1, num_unsup_rows+num_sup_rows+1):
        x2.append(i)

    fig, ax1 = plt.subplots()

    ax1.plot(x, df_whole['within ' + str(f)], color="limegreen", label='within ' + str(f))
    ax1.plot(x, df_whole['within ' + str(s)], color="green", label='within ' + str(s))
    ax1.plot(x, df_whole['between'], color="blue", label="between")

    ax1.axvline(x=0, color='r', linestyle='--')
    ax1.axvline(x=num_unsup_rows, color='r', linestyle='--')
    ax1.set_ylabel('Distance')
    ax1.axvline(x=time_step, color='black', linestyle='dashed')
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(x2, accs, color="coral", label="accuracy")
    ax2.set_ylabel('Accuracy')


    plt.xlabel("Epoch")
    plt.title(str(f) + " " + str(s) + " distances and accuracy across batches")
    plt.savefig(f"{loc}/"+str(time_step) + " Acc distance plot 0 1.png")
    plt.show()

def plot_skip_batch(tup, skip_batch, time_step, sup_df, loc):

    f, s = tup
    df_no_train = pd.read_csv("no_train_0_1.csv")

    df_unsup = pd.read_csv("Unsup 1 0 lr=0.05")
    unsup_rows = []

    for i in range(len(df_unsup["between"])):
        if i % skip_batch == 0:
            unsup_rows.append(df_unsup.loc[i])

    df_unsup = pd.DataFrame(unsup_rows)
    num_unsup_rows = len(unsup_rows)

    df_sup = pd.read_csv(sup_df)
    sup_rows = []
    for i in range(len(df_sup["between"])):
        if i % skip_batch == 0:
            sup_rows.append(df_sup.loc[i])
    df_sup = pd.DataFrame(sup_rows)
    num_sup_rows = len(sup_rows)

    accs = df_sup["Accuracy"].apply(scale)
    df_sup.drop(columns=["Accuracy"], inplace=True)

    df_whole = pd.concat([df_no_train, df_unsup, df_sup])
    x = []
    for i in range(1 + num_unsup_rows + num_sup_rows):
        x.append(i)
    x2 = []
    for i in range(num_unsup_rows + 1, num_unsup_rows + num_sup_rows + 1):
        x2.append(i)

    fig, ax1 = plt.subplots()

    ax1.plot(x, df_whole['within ' + str(f)], color="limegreen", label='within ' + str(f))
    ax1.plot(x, df_whole['within ' + str(s)], color="green", label='within ' + str(s))
    ax1.plot(x, df_whole['between'], color="blue", label="between")

    ax1.axvline(x=0, color='r', linestyle='--')
    ax1.axvline(x=num_unsup_rows, color='r', linestyle='--')
    ax1.set_ylabel('Distance')
    ax1.axvline(x=time_step, color='black', linestyle='dashed')
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(x2, accs, color="coral", label="accuracy")
    ax2.set_ylabel('Accuracy')

    plt.xlabel("Epoch")
    plt.title(str(f) + " " + str(s) + " distances and accuracy across batches")
    plt.savefig(f"{loc}/" + str(time_step) + " Acc distance plot 0 1.png")
    plt.show()



if __name__ == '__main__':

    batch_size = 100
    lr = 0.0001

    #train_unsup(tup=(1,0), batch_size=batch_size, lr=0.05)
    '''
    unsup_model = neural_networks.AutoEncoder()
    unsup_model.load_state_dict(torch.load("unsup_weights/599 100 0.05 unsup_model.pth"))
    train_sup(unsup_model, tup=(1,0), batch_size=batch_size, lr=lr)

    '''
    for i in range(0, 121):
        plot_skip_batch(tup=(1,0), time_step=i, sup_df="Sup Slowed down 1 0 lr=0.005", loc="new_plots/blue-green/fast_lr", skip_batch=10)
    
