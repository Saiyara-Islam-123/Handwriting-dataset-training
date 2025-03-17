import neural_networks
from torch import optim
import torch.nn as nn
from sampling import *
import plotting_X
import matplotlib.pyplot as plt
from similarity_matrix_mnist import *
import numpy as np
import pandas as pd
from accuracy import *
import json
from plot_custom_label import *


def plot(dict_loss, dict_acc, is_sup):
    plt.xlim(0, 10)  # Set x-axis range from 2 to 8
    if is_sup == "sup":

        plt.plot(dict_loss.keys(), dict_loss.values(), label='Loss on train data', marker='o',
                 color='red')  # First line with markers

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Train loss")
        plt.legend()
        plt.savefig("Train loss. 10 epochs each " + str(is_sup) + " .png")

        plt.show()

        plt.plot(dict_acc.keys(), dict_acc.values(), label='Accuracy on test data', marker='s',
                 color='pink')  # First line with markers

        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Test accuracy")
        plt.legend()
        plt.savefig("Test Accuracy. 10 epochs each " + str(is_sup) + " .png")

        plt.show()


    else:
        plt.plot(dict_loss.keys(), dict_loss.values(), label='Loss on train data', marker='o',
                 color='red')

        plt.xlabel("Epochs")
        plt.ylabel("Accuracy or loss")
        plt.title("Train Loss")
        plt.legend()
        plt.savefig("Train loss. 10 epochs each " + str(is_sup) + " .png")

        plt.show()


if __name__ == "__main__":
    # dict_unsup_epoch_to_upper_triangle = {}
    torch.manual_seed(0)

    X = create_dataset.get_inputs()  # I wanna sample from here
    y = create_dataset.get_labels()

    pair_avg_distances = {}
    pair_avg_distances[(4, 4)] = [sampled_avg_distance((4, 4), X, y)]
    pair_avg_distances[(4, 9)] = [sampled_avg_distance((4, 9), X, y)]
    pair_avg_distances[(9, 9)] = [sampled_avg_distance((9, 9), X, y)]

    pair_avg_distances[(1, 1)] = [sampled_avg_distance((1, 1), X, y)]
    pair_avg_distances[(1, 0)] = [sampled_avg_distance((1, 0), X, y)]
    pair_avg_distances[(0, 0)] = [sampled_avg_distance((0, 0), X, y)]

    pair_avg_distances[(4, 0)] = [sampled_avg_distance((4, 0), X, y)]
    pair_avg_distances[(9, 1)] = [sampled_avg_distance((9, 1), X, y)]

    pair_avg_distances[(4, 1)] = [sampled_avg_distance((4, 1), X, y)]
    pair_avg_distances[(9, 0)] = [sampled_avg_distance((9, 0), X, y)]

    # print(X_filtered.shape)

    # plotting_X.plot(X_filtered.reshape(X_filtered.shape[0], 784), y_filtered, -1, "pre-training")

    dict_unsup_epoch_loss = {}
    dict_sup_epoch_loss = {}
    dict_sup_epoch_acc = {}

    model = neural_networks.AutoEncoder()
    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)

    inputs = create_dataset.get_inputs()

    batches = list(torch.split(inputs, 32))
    batches_of_labels = list(torch.split(create_dataset.get_labels(), 32))

    # pairwise_similarity_matrix = compute_pairwise_similarities(X.numpy(), y.numpy())
    # plot_similarity_matrix(pairwise_similarity_matrix, str(-1) + " pretraining")

    # dict_unsup_epoch_to_upper_triangle[-1] = upper_triangle(pairwise_similarity_matrix)

    loss_dict = {}

    print("\nUnsupervised part!")
    for epoch in range(2):

        encoder_outputs_list = []
        labels_list = []

        for i in range(len(batches)):
            optimizer.zero_grad()

            outputs = model(batches[i])
            encoder_outputs_list.append(model.encoded.detach().numpy())
            labels_list.append(batches_of_labels[i].detach().numpy())

            loss = loss_fn(outputs, batches[i])
            loss.backward()
            optimizer.step()

            if len(encoder_outputs_list) % 10000 == 0:
                print(len(encoder_outputs_list))

                intermediate_output_list_unsup = np.array(
                    [item for sublist in encoder_outputs_list for item in sublist])
                intermediate_labels_list_unsup = np.array([item for sublist in labels_list for item in sublist])

                pair_avg_distances[(4, 4)] = pair_avg_distances[(4, 4)] + [
                    sampled_avg_distance((4, 4), torch.tensor(intermediate_output_list_unsup),
                                         torch.Tensor(intermediate_labels_list_unsup))]
                pair_avg_distances[(4, 9)] = pair_avg_distances[(4, 9)] + [
                    sampled_avg_distance((4, 9), torch.tensor(intermediate_output_list_unsup),
                                         torch.Tensor(intermediate_labels_list_unsup))]
                pair_avg_distances[(9, 9)] = pair_avg_distances[(9, 9)] + [
                    sampled_avg_distance((9, 9), torch.tensor(intermediate_output_list_unsup),
                                         torch.Tensor(intermediate_labels_list_unsup))]

                pair_avg_distances[(1, 1)] = pair_avg_distances[(1, 1)] + [
                    sampled_avg_distance((1, 1), torch.tensor(intermediate_output_list_unsup),
                                         torch.Tensor(intermediate_labels_list_unsup))]
                pair_avg_distances[(1, 0)] = pair_avg_distances[(1, 0)] + [
                    sampled_avg_distance((1, 0), torch.tensor(intermediate_output_list_unsup),
                                         torch.Tensor(intermediate_labels_list_unsup))]
                pair_avg_distances[(0, 0)] = pair_avg_distances[(0, 0)] + [
                    sampled_avg_distance((0, 0), torch.tensor(intermediate_output_list_unsup),
                                         torch.Tensor(intermediate_labels_list_unsup))]

                pair_avg_distances[(4, 0)] = pair_avg_distances[(4, 0)] + [
                    sampled_avg_distance((4, 0), torch.tensor(intermediate_output_list_unsup),
                                         torch.Tensor(intermediate_labels_list_unsup))]
                pair_avg_distances[(4, 1)] = pair_avg_distances[(4, 1)] + [
                    sampled_avg_distance((4, 1), torch.tensor(intermediate_output_list_unsup),
                                         torch.Tensor(intermediate_labels_list_unsup))]
                pair_avg_distances[(9, 0)] = pair_avg_distances[(9, 0)] + [
                    sampled_avg_distance((9, 0), torch.tensor(intermediate_output_list_unsup),
                                         torch.Tensor(intermediate_labels_list_unsup))]

                pair_avg_distances[(9, 1)] = pair_avg_distances[(9, 1)] + [
                    sampled_avg_distance((9, 1), torch.tensor(intermediate_output_list_unsup),
                                         torch.Tensor(intermediate_labels_list_unsup))]

        # print(outputs_list_flattened.shape)
        # print(labels_list_flattened.shape)

        # pairwise_similarity_matrix = compute_pairwise_similarities(outputs_list_flattened, labels_list_flattened)
        # plot_similarity_matrix(pairwise_similarity_matrix, str(epoch) + " unsup, 5 epochs")

        # outputs_filtered, y_filtered = plotting_X.filter(torch.tensor(outputs_list_flattened), y, [1, 0, 4, 9])
        # plotting_X.plot(outputs_filtered, y_filtered, epoch, "unsup")

        # dict_unsup_epoch_to_upper_triangle[epoch] = upper_triangle(pairwise_similarity_matrix)
        # dict_unsup_epoch_loss[epoch] = loss.item()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

    # unsup_output_dict = {"unsup outputs": outputs_list_flattened.tolist()}
    # with open("unsup.json", "w") as f:
    # json.dump(unsup_output_dict, f)

    model_2 = neural_networks.LastLayer(model)

    model_2.train()

    loss_fn_2 = nn.CrossEntropyLoss()
    optimizer_2 = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)

    print("\nSupervised part!")
    dict_sup_epoch_to_upper_triangle = {}

    ###################################################################################################

    for epoch in range(2):
        outputs_autoencoder_list = []
        labels_autoencoder_list = []
        for i in range(len(batches)):
            optimizer_2.zero_grad()

            outputs_supervised = model_2(batches[i])
            outputs_autoencoder_list.append(model_2.autoencoder_output.detach().numpy())
            labels_autoencoder_list.append(batches_of_labels[i].detach().numpy())

            loss = loss_fn_2(outputs_supervised, batches_of_labels[i])

            loss.backward()
            optimizer_2.step()

            if len(outputs_autoencoder_list) % 10000 == 0:
                print(len(outputs_autoencoder_list))

                intermediate_output_list_sup = np.array(
                    [item for sublist in outputs_autoencoder_list for item in sublist])
                intermediate_labels_list_sup = np.array(
                    [item for sublist in labels_autoencoder_list for item in sublist])

                pair_avg_distances[(4, 4)] = pair_avg_distances[(4, 4)] + [
                    sampled_avg_distance((4, 4), torch.tensor(intermediate_output_list_sup),
                                         torch.Tensor(intermediate_labels_list_sup))]
                pair_avg_distances[(4, 9)] = pair_avg_distances[(4, 9)] + [
                    sampled_avg_distance((4, 9), torch.tensor(intermediate_output_list_sup),
                                         torch.Tensor(intermediate_labels_list_sup))]
                pair_avg_distances[(9, 9)] = pair_avg_distances[(9, 9)] + [
                    sampled_avg_distance((9, 9), torch.tensor(intermediate_output_list_sup),
                                         torch.Tensor(intermediate_labels_list_sup))]

                pair_avg_distances[(1, 1)] = pair_avg_distances[(1, 1)] + [
                    sampled_avg_distance((1, 1), torch.tensor(intermediate_output_list_sup),
                                         torch.Tensor(intermediate_labels_list_sup))]
                pair_avg_distances[(1, 0)] = pair_avg_distances[(1, 0)] + [
                    sampled_avg_distance((1, 0), torch.tensor(intermediate_output_list_sup),
                                         torch.Tensor(intermediate_labels_list_sup))]
                pair_avg_distances[(0, 0)] = pair_avg_distances[(0, 0)] + [
                    sampled_avg_distance((0, 0), torch.tensor(intermediate_output_list_sup),
                                         torch.Tensor(intermediate_labels_list_sup))]

                pair_avg_distances[(4, 0)] = pair_avg_distances[(4, 0)] + [
                    sampled_avg_distance((4, 0), torch.tensor(intermediate_output_list_sup),
                                         torch.Tensor(intermediate_labels_list_sup))]
                pair_avg_distances[(4, 1)] = pair_avg_distances[(4, 1)] + [
                    sampled_avg_distance((4, 1), torch.tensor(intermediate_output_list_sup),
                                         torch.Tensor(intermediate_labels_list_sup))]
                pair_avg_distances[(9, 0)] = pair_avg_distances[(9, 0)] + [
                    sampled_avg_distance((9, 0), torch.tensor(intermediate_output_list_sup),
                                         torch.Tensor(intermediate_labels_list_sup))]

                pair_avg_distances[(9, 1)] = pair_avg_distances[(9, 1)] + [
                    sampled_avg_distance((9, 1), torch.tensor(intermediate_output_list_sup),
                                         torch.Tensor(intermediate_labels_list_sup))]

        # pairwise_similarity_matrix = compute_pairwise_similarities(outputs_autoencoder_flattened, labels_autoencoder_flattened)
        # plot_similarity_matrix(pairwise_similarity_matrix, str(epoch) + " sup, 5 epochs")

        # outputs_sup_filtered, y_filtered_sup = plotting_X.filter(torch.tensor(outputs_autoencoder_flattened), y, [1, 0, 4, 9])
        # plotting_X.plot(outputs_sup_filtered, y_filtered_sup, epoch, "sup")

        # dict_sup_epoch_to_upper_triangle[epoch] = upper_triangle(pairwise_similarity_matrix)
        # dict_sup_epoch_loss[epoch] = loss.item()
        # dict_sup_epoch_acc[epoch] = acc(model_2)

        print(f"Epoch {epoch}, Loss: {loss.item()}")

        # print(model_2.autoencoder_output.shape)

    # sup_output_dict = {"sup outputs": outputs_autoencoder_flattened.tolist()}
    # with open("sup.json", "w") as f:
    # json.dump(sup_output_dict, f)

    within_4 = pair_avg_distances[(4, 4)]
    within_9 = pair_avg_distances[(9, 9)]
    between_4_9 = pair_avg_distances[(4, 9)]

    within_1 = pair_avg_distances[(1, 1)]
    within_0 = pair_avg_distances[(0, 0)]
    between_1_0 = pair_avg_distances[(1, 0)]

    between_4_0 = pair_avg_distances[(4, 0)]
    between_9_0 = pair_avg_distances[(9, 0)]

    between_4_1 = pair_avg_distances[(4, 1)]
    between_9_1 = pair_avg_distances[(9, 1)]

    plt.plot(list(range(len(within_4))), within_4, label='Within 4', marker='o',
             color='green')  # First line with markers
    plt.plot(list(range(len(within_9))), within_9, label='Within 9', marker='s',
             color='green')  # First line with markers
    plt.plot(list(range(len(between_4_9))), between_4_9, label='Between 4 and 9', marker='x',
             color="blue")  # Second line with different markers

    plt.plot(list(range(len(within_1))), within_1, label='Within 1', marker='v',
             color='green')  # First line with markers
    plt.plot(list(range(len(within_0))), within_0, label='Within 0', marker=',',
             color='green')  # First line with markers
    plt.plot(list(range(len(between_1_0))), between_1_0, label='Between 1 and 0', marker='^',
             color="blue")  # Second line with different markers

    plt.plot(list(range(len(between_4_0))), between_4_0, label='Between 4 and 0', marker='*', color="blue")
    plt.plot(list(range(len(between_4_1))), between_4_1, label='Between 4 and 1', marker='h', color="blue")

    plt.plot(list(range(len(between_9_0))), between_9_0, label='Between 9 and 0', marker='D', color="blue")
    plt.plot(list(range(len(between_9_1))), between_9_1, label='Between 9 and 1', marker='|', color="blue")

    plt.xlabel("Phases of training")
    plt.ylabel("Distances")
    plt.title("1,0,4,9")
    plt.legend()
    plt.savefig("Average of averages distances, 1,0,4,9, batches of 1.png")

    plt.show()

    # measuring accuracy

    # df_unsup_triangle = pd.DataFrame(dict_unsup_epoch_to_upper_triangle)
    # df_sup_triangle = pd.DataFrame(dict_sup_epoch_to_upper_triangle)
    # df_unsup_triangle.to_csv('unsup_triangle 5 epochs each.csv', index=False)
    # df_sup_triangle.to_csv('sup_triangle.csv 5 epochs each.csv', index=False)

    # plotting
    # plot(dict_unsup_epoch_loss, {}, "unsup")
    # plot(dict_sup_epoch_loss, dict_sup_epoch_acc, "sup")

