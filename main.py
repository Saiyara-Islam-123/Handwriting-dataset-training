
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
        #dict_unsup_epoch_to_upper_triangle = {}
        torch.manual_seed(0)

        X = create_dataset.get_inputs() #I wanna sample from here
        y = create_dataset.get_labels()

        pair_avg_distances = {}
        pair_avg_distances[(4,4)] = [sampled_avg_distance((4, 4), X , y)]
        pair_avg_distances[(4, 9)] = [sampled_avg_distance((4, 9), X, y)]
        pair_avg_distances[(9, 9)] = [sampled_avg_distance((9, 9), X, y)]


        #print(X_filtered.shape)

        #plotting_X.plot(X_filtered.reshape(X_filtered.shape[0], 784), y_filtered, -1, "pre-training")

        dict_unsup_epoch_loss = {}
        dict_sup_epoch_loss = {}
        dict_sup_epoch_acc = {}

        model = neural_networks.AutoEncoder()
        loss_fn = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)

        inputs = create_dataset.get_inputs()



        batches = list(torch.split(inputs, 1))
        batches_of_labels = list(torch.split(create_dataset.get_labels(), 1))


        #pairwise_similarity_matrix = compute_pairwise_similarities(X.numpy(), y.numpy())
        #plot_similarity_matrix(pairwise_similarity_matrix, str(-1) + " pretraining")

        #dict_unsup_epoch_to_upper_triangle[-1] = upper_triangle(pairwise_similarity_matrix)

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



            outputs_list_flattened = np.array([item for sublist in encoder_outputs_list for item in sublist])
            labels_list_flattened = np.array([item for sublist in labels_list for item in sublist])




            #print(outputs_list_flattened.shape)
            #print(labels_list_flattened.shape)

            #pairwise_similarity_matrix = compute_pairwise_similarities(outputs_list_flattened, labels_list_flattened)
            #plot_similarity_matrix(pairwise_similarity_matrix, str(epoch) + " unsup, 5 epochs")

            #outputs_filtered, y_filtered = plotting_X.filter(torch.tensor(outputs_list_flattened), y, [1, 0, 4, 9])
            #plotting_X.plot(outputs_filtered, y_filtered, epoch, "unsup")



            pair_avg_distances[(4, 4)] = pair_avg_distances[(4, 4)] + [sampled_avg_distance((4, 4), torch.tensor(outputs_list_flattened), y)]
            pair_avg_distances[(4, 9)] = pair_avg_distances[(4, 9)] + [sampled_avg_distance((4, 9), torch.tensor(outputs_list_flattened), y)]
            pair_avg_distances[(9, 9)] = pair_avg_distances[(9, 9)] + [sampled_avg_distance((9, 9), torch.tensor(outputs_list_flattened), y)]


            #dict_unsup_epoch_to_upper_triangle[epoch] = upper_triangle(pairwise_similarity_matrix)
            #dict_unsup_epoch_loss[epoch] = loss.item()

            print(f"Epoch {epoch}, Loss: {loss.item()}")

        #unsup_output_dict = {"unsup outputs": outputs_list_flattened.tolist()}
        #with open("unsup.json", "w") as f:
            #json.dump(unsup_output_dict, f)

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

            outputs_autoencoder_flattened = np.array([item for sublist in outputs_autoencoder_list for item in sublist])
            labels_autoencoder_flattened = np.array([item for sublist in labels_autoencoder_list for item in sublist])




            #pairwise_similarity_matrix = compute_pairwise_similarities(outputs_autoencoder_flattened, labels_autoencoder_flattened)
            #plot_similarity_matrix(pairwise_similarity_matrix, str(epoch) + " sup, 5 epochs")


            #outputs_sup_filtered, y_filtered_sup = plotting_X.filter(torch.tensor(outputs_autoencoder_flattened), y, [1, 0, 4, 9])
            #plotting_X.plot(outputs_sup_filtered, y_filtered_sup, epoch, "sup")

            #dict_sup_epoch_to_upper_triangle[epoch] = upper_triangle(pairwise_similarity_matrix)
            #dict_sup_epoch_loss[epoch] = loss.item()
            #dict_sup_epoch_acc[epoch] = acc(model_2)

            print(f"Epoch {epoch}, Loss: {loss.item()}")

            #print(model_2.autoencoder_output.shape)

            pair_avg_distances[(4, 4)] = pair_avg_distances[(4, 4)] + [sampled_avg_distance((4, 4), torch.tensor(outputs_autoencoder_flattened), y)]
            pair_avg_distances[(4, 9)] = pair_avg_distances[(4, 9)] + [sampled_avg_distance((4, 9), torch.tensor(outputs_autoencoder_flattened), y)]
            pair_avg_distances[(9, 9)] = pair_avg_distances[(9, 9)] + [sampled_avg_distance((9, 9), torch.tensor(outputs_autoencoder_flattened), y)]

        #sup_output_dict = {"sup outputs": outputs_autoencoder_flattened.tolist()}
        #with open("sup.json", "w") as f:
            #json.dump(sup_output_dict, f)


        within_4= pair_avg_distances[(4,4)]
        within_9 = pair_avg_distances[(9, 9)]
        between = pair_avg_distances[(4, 9)]

        
        plt.xlim(0, 6)  # Set x-axis range from 2 to 8
        plt.ylim(0, 10)

        plt.plot(list(range(len(within_4))), within_4, label='Within 4', marker='o', color='green')  # First line with markers
        plt.plot(list(range(len(within_9))), within_9, label='Within 9', marker='s',color='green')  # First line with markers
        plt.plot(list(range(len(between))), between, label='Between 4 and 9', marker='x', color = "blue")  # Second line with different markers
    
        plt.xlabel("Phases of training")
        plt.ylabel("Distances")
        plt.title("Within and between 4 and 9")
        plt.legend()
        plt.savefig("Average of averages distance between 4 and 9 and within 4 and 9, seeds 0, 10 epochs each, batch of 1")

        plt.show()
        
        

        #measuring accuracy



        #df_unsup_triangle = pd.DataFrame(dict_unsup_epoch_to_upper_triangle)
        #df_sup_triangle = pd.DataFrame(dict_sup_epoch_to_upper_triangle)
        #df_unsup_triangle.to_csv('unsup_triangle 5 epochs each.csv', index=False)
        #df_sup_triangle.to_csv('sup_triangle.csv 5 epochs each.csv', index=False)


        #plotting
        #plot(dict_unsup_epoch_loss, {}, "unsup")
        #plot(dict_sup_epoch_loss, dict_sup_epoch_acc, "sup")


    
