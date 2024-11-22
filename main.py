
import neural_networks2
from torch import optim
import torch.nn as nn
from sampling import *
#import plotting_X
import matplotlib.pyplot as plt



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

        model = neural_networks2.AutoEncoder()
        loss_fn = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)

        inputs = create_dataset.get_inputs()



        batches = list(torch.split(inputs, 64))
        batches_of_labels = list(torch.split(create_dataset.get_labels(), 64))




        print("\nUnsupervised part!")
        for epoch in range(5):

            outputs_list = []
            for batch in batches:
                optimizer.zero_grad()

                outputs = model(batch)
                outputs_list.append(outputs)


                loss = loss_fn(outputs, batch)
                loss.backward()
                optimizer.step()

            #outputs_filtered, y_filtered = plotting_X.filter(torch.cat(outputs_list), y, [1, 0, 4, 9])
            #plotting_X.plot(outputs_filtered, y_filtered, epoch, "unsup")

            pair_avg_distances[(4, 4)] = pair_avg_distances[(4, 4)] + [sampled_avg_distance((4, 4), torch.cat(outputs_list, dim=0), y)]
            pair_avg_distances[(4, 9)] = pair_avg_distances[(4, 9)] + [sampled_avg_distance((4, 9), torch.cat(outputs_list, dim=0), y)]
            pair_avg_distances[(9, 9)] = pair_avg_distances[(9, 9)] + [sampled_avg_distance((9, 9), torch.cat(outputs_list, dim=0), y)]

            print(f"Epoch {epoch+1}, Loss: {loss.item()}")



        model_2 = neural_networks2.LastLayer(model)

        model_2.train()

        loss_fn_2 = nn.CrossEntropyLoss()
        optimizer_2 = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)


        print("\nSupervised part!")

        ###################################################################################################

        for epoch in range(5):
            outputs_autoencoder_list = []
            for i in range(len(batches)):
                optimizer_2.zero_grad()

                outputs_supervised = model_2(batches[i])
                outputs_autoencoder_list.append(model_2.autoencoder_output)


                loss = loss_fn_2(outputs_supervised, batches_of_labels[i])

                loss.backward()
                optimizer_2.step()

            #outputs_sup_filtered, y_filtered_sup = plotting_X.filter(torch.cat(outputs_autoencoder_list), y, [1, 0, 4, 9])
            #plotting_X.plot(outputs_sup_filtered, y_filtered_sup, epoch, "sup")

            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

            #print(model_2.autoencoder_output.shape)

            pair_avg_distances[(4, 4)] = pair_avg_distances[(4, 4)] + [sampled_avg_distance((4, 4), torch.cat(outputs_autoencoder_list, dim=0), y)]
            pair_avg_distances[(4, 9)] = pair_avg_distances[(4, 9)] + [sampled_avg_distance((4, 9), torch.cat(outputs_autoencoder_list, dim=0), y)]
            pair_avg_distances[(9, 9)] = pair_avg_distances[(9, 9)] + [sampled_avg_distance((9, 9), torch.cat(outputs_autoencoder_list, dim=0), y)]

        within_1= pair_avg_distances[(4,4)]
        within_0 = pair_avg_distances[(9, 9)]
        between = pair_avg_distances[(4, 9)]


        plt.xlim(0, 11)  # Set x-axis range from 2 to 8
        plt.ylim(0, 5)

        plt.plot(list(range(len(within_1))), within_1, label='Within 4', marker='o', color='green')  # First line with markers
        plt.plot(list(range(len(within_0))), within_0, label='Within 9', marker='s',color='green')  # First line with markers
        plt.plot(list(range(len(between))), between, label='Between 4 and 9', marker='x', color = "blue")  # Second line with different markers
    
        plt.xlabel("Phases")
        plt.ylabel("Distances")
        plt.title("Distance between 4 and 9 and within 4 before, during and after both training CNN")
        plt.savefig("Average of averages distance between 4 and 9 and within 4 before, during and after both training CNN trial 2, seed, neural networks 2")
        plt.show()


        #measuring accuracy




        test_inputs = create_dataset.get_test_inputs()
        test_labels = create_dataset.get_test_labels()

        pred_labels = model_2(test_inputs).argmax(dim=1)

        acc = (pred_labels == test_labels).float().mean().item()

        print("\nAccuracy = ")
        print(acc * 100)
    
