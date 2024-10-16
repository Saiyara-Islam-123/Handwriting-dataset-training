
import neural_networks
from torch import optim
import torch.nn as nn
from plotting_X import *
from sampling import *



if __name__ == "__main__":

    X = create_dataset.get_inputs()
    y = create_dataset.get_labels()

    pair_avg_distances = {}
    pair_avg_distances[(1,1)] = [sampled_avg_distance((1, 1), X , y)]
    pair_avg_distances[(1, 0)] = [sampled_avg_distance((1, 0), X, y)]
    pair_avg_distances[(0, 0)] = [sampled_avg_distance((0, 0), X, y)]

    #X_filtered, y_filtered = plotting_X.filter(X, y, [1, 0, 4, 9])

    #print(X_filtered.shape)

    #plotting_X.plot(X_filtered.reshape(X_filtered.shape[0], 784), y_filtered, -1)
    
    model = neural_networks.AutoEncoder()
    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    inputs = create_dataset.get_inputs()
    
    batches = list(torch.split(inputs, 64))
    batches_of_labels = list(torch.split(create_dataset.get_labels(), 64))


    print("\nUnsupervised part!")
    for epoch in range(10):

        outputs_list = []
        for batch in batches:
            optimizer.zero_grad()
            #print(batch.shape)

            outputs = model(batch)
            outputs_list.append(outputs)


            #print(outputs.shape)

            loss = loss_fn(outputs, batch)
            loss.backward()
            optimizer.step()

        #outputs_filtered, y_filtered = filter(torch.cat(outputs_list), y, [1, 0, 4, 9])
        #plot(outputs_filtered, y_filtered, epoch, "unsup")

        pair_avg_distances[(1, 1)] = pair_avg_distances[(1, 1)] + [sampled_avg_distance((1, 1), torch.cat(outputs_list, dim=0), y)]
        pair_avg_distances[(1, 0)] = pair_avg_distances[(1, 0)] + [sampled_avg_distance((1, 0), torch.cat(outputs_list, dim=0), y)]
        pair_avg_distances[(0, 0)] = pair_avg_distances[(0, 0)] + [sampled_avg_distance((0, 0), torch.cat(outputs_list, dim=0), y)]
            
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


    model_2 = neural_networks.LastLayer(model)  

    model_2.train()
    
    loss_fn_2 = nn.CrossEntropyLoss()
    optimizer_2 = optim.Adam(model.parameters(), lr=0.01)

    print("\nSupervised part!")

    ###################################################################################################

    for epoch in range(10):
        outputs_supervised_list = []
        for i in range(len(batches)):
            optimizer_2.zero_grad()

            #print(batches[i].shape)
            outputs_supervised = model_2(batches[i])

            outputs_supervised_list.append(outputs_supervised)

            #print(outputs_supervised.shape)



            loss = loss_fn_2(outputs_supervised, batches_of_labels[i])

            loss.backward()
            optimizer_2.step()

        #outputs_sup_filtered, y_filtered_sup = filter(torch.cat(outputs_supervised_list), y, [1, 0, 4, 9])
        #plot(outputs_sup_filtered, y_filtered_sup, epoch, "sup")

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        pair_avg_distances[(1, 1)] = pair_avg_distances[(1, 1)] + [sampled_avg_distance((1, 1), torch.cat(outputs_supervised_list, dim=0), y)]
        pair_avg_distances[(1, 0)] = pair_avg_distances[(1, 0)] + [sampled_avg_distance((1, 0), torch.cat(outputs_supervised_list, dim=0), y)]
        pair_avg_distances[(0, 0)] = pair_avg_distances[(0, 0)] + [sampled_avg_distance((0, 0), torch.cat(outputs_supervised_list, dim=0), y)]

    within_1 = pair_avg_distances[(1,1)]
    within_0 = pair_avg_distances[(0, 0)]
    between = pair_avg_distances[(1,0)]



    plt.plot(list(range(len(within_1))), within_1, label='Within 1', marker='o', color='green')  # First line with markers
    plt.plot(list(range(len(within_0))), within_0, label='Within 0', marker='s',color='green')  # First line with markers
    plt.plot(list(range(len(between))), between, label='Between 1 and 0', marker='x', color = "blue")  # Second line with different markers

    plt.xlabel("Phases")
    plt.ylabel("Distances")
    plt.title("Distance between 1 and 0 and within 1 before, during and after both training CNN")
    plt.savefig("Distance between 1 and 0 and within 1 before, during and after both training CNN trial 2.png")
    plt.show()
    

    #measuring accuracy



    test_inputs = create_dataset.get_test_inputs()
    test_labels = create_dataset.get_test_labels()
    
    pred_labels = model_2(test_inputs).argmax(dim=1)
    
    acc = (pred_labels == test_labels).float().mean().item()
    
    print("\nAccuracy = ")
    print(acc * 100)
    

