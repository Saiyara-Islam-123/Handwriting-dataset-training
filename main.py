
import neural_networks
from torch import optim
import torch.nn as nn
from sampling import *
#import plotting_X
import matplotlib.pyplot as plt
from similarity_matrix_mnist import *
import numpy as np



if __name__ == "__main__":

        torch.manual_seed(0)

        X = create_dataset.get_inputs()
        y = create_dataset.get_labels()

        pair_avg_distances = {}
        pair_avg_distances[(4,4)] = [sampled_avg_distance((4, 4), X , y)]
        pair_avg_distances[(4, 9)] = [sampled_avg_distance((4, 9), X, y)]
        pair_avg_distances[(9, 9)] = [sampled_avg_distance((9, 9), X, y)]


        #X_filtered, y_filtered = plotting_X.filter(X, y, [1, 0, 4, 9])

        #print(X_filtered.shape)

        #plotting_X.plot(X_filtered.reshape(X_filtered.shape[0], 784), y_filtered, -1, "pre-training")

        model = neural_networks.AutoEncoder()
        loss_fn = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)

        inputs = create_dataset.get_inputs()



        batches = list(torch.split(inputs, 64))
        batches_of_labels = list(torch.split(create_dataset.get_labels(), 64))

        #pairwise_similarity_matrix = compute_pairwise_similarities(X.numpy(), y.numpy())
        #plot_similarity_matrix(pairwise_similarity_matrix, str(-1) + " pretraining")


        print("\nUnsupervised part!")
        for epoch in range(10):

            outputs_list = []
            labels_list = []


            for i in range(len(batches)):
                optimizer.zero_grad()

                outputs = model(batches[i])
                outputs_list.append(outputs.detach().numpy())
                labels_list.append(batches_of_labels[i].detach().numpy())

                loss = loss_fn(outputs, batches[i])
                loss.backward()
                optimizer.step()



            outputs_list_flattened = np.array([item for sublist in outputs_list for item in sublist])
            labels_list_flattened = np.array([item for sublist in labels_list for item in sublist])

            #print(outputs_list_flattened.shape)
            #print(labels_list_flattened.shape)

            #pairwise_similarity_matrix = compute_pairwise_similarities(outputs_list_flattened, labels_list_flattened)
            #plot_similarity_matrix(pairwise_similarity_matrix, str(epoch) + " unsup")

            #outputs_filtered, y_filtered = plotting_X.filter(torch.cat(outputs_list), y, [1, 0, 4, 9])
            #plotting_X.plot(outputs_filtered, y_filtered, epoch, "unsup")



            pair_avg_distances[(4, 4)] = pair_avg_distances[(4, 4)] + [sampled_avg_distance((4, 4), torch.tensor(outputs_list_flattened), y)]
            pair_avg_distances[(4, 9)] = pair_avg_distances[(4, 9)] + [sampled_avg_distance((4, 9), torch.tensor(outputs_list_flattened), y)]
            pair_avg_distances[(9, 9)] = pair_avg_distances[(9, 9)] + [sampled_avg_distance((9, 9), torch.tensor(outputs_list_flattened), y)]

            print(f"Epoch {epoch+1}, Loss: {loss.item()}")



        model_2 = neural_networks.LastLayer(model)

        model_2.train()

        loss_fn_2 = nn.CrossEntropyLoss()
        optimizer_2 = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)


        print("\nSupervised part!")

        ###################################################################################################

        for epoch in range(10):
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
            #plot_similarity_matrix(pairwise_similarity_matrix, str(epoch) + " sup")


            #outputs_sup_filtered, y_filtered_sup = plotting_X.filter(torch.cat(outputs_autoencoder_list), y, [1, 0, 4, 9])
            #plotting_X.plot(outputs_sup_filtered, y_filtered_sup, epoch, "sup")

            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

            #print(model_2.autoencoder_output.shape)

            pair_avg_distances[(4, 4)] = pair_avg_distances[(4, 4)] + [sampled_avg_distance((4, 4), torch.tensor(outputs_autoencoder_flattened), y)]
            pair_avg_distances[(4, 9)] = pair_avg_distances[(4, 9)] + [sampled_avg_distance((4, 9), torch.tensor(outputs_autoencoder_flattened), y)]
            pair_avg_distances[(9, 9)] = pair_avg_distances[(9, 9)] + [sampled_avg_distance((9, 9), torch.tensor(outputs_autoencoder_flattened), y)]

        within_4= pair_avg_distances[(4,4)]
        within_9 = pair_avg_distances[(9, 9)]
        between = pair_avg_distances[(4, 9)]


        plt.xlim(0, 21)  # Set x-axis range from 2 to 8
        plt.ylim(0, 10)

        plt.plot(list(range(len(within_4))), within_4, label='Within 4', marker='o', color='green')  # First line with markers
        plt.plot(list(range(len(within_9))), within_9, label='Within 9', marker='s',color='green')  # First line with markers
        plt.plot(list(range(len(between))), between, label='Between 4 and 9', marker='x', color = "blue")  # Second line with different markers
    
        plt.xlabel("Phases")
        plt.ylabel("Distances")
        plt.title("Within and between 4 and 9")
        plt.savefig("Average of averages distance between 4 and 9 and within 4 and 9, seed 0, 10 epochs each")
        plt.legend()
        plt.show()


        #measuring accuracy

        test_inputs = create_dataset.get_test_inputs()
        test_labels = create_dataset.get_test_labels()

        pred_labels = model_2(test_inputs).argmax(dim=1)

        acc = (pred_labels == test_labels).float().mean().item()

        print("\nAccuracy = ")
        print(acc * 100)



    
