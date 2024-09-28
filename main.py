import create_dataset
import neural_networks
from torch import optim
import torch.nn as nn
import torch
import plotting_X
import random

if __name__ == "__main__":

    
    model = neural_networks.AutoEncoder()
    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    inputs = create_dataset.get_inputs()
    
    batches = list(torch.split(inputs, 64))


    print("\nUnsupervised part!")
    for epoch in range(10):
        
        for batch in batches:
            optimizer.zero_grad()
            outputs = model(batch)
            #print(outputs.shape)

            loss = loss_fn(outputs, batch)
            loss.backward()
            optimizer.step()
            
            
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    model_2 = neural_networks.LastLayer(model)  

    model_2.train()
    
    loss_fn_2 = nn.CrossEntropyLoss()
    optimizer_2 = optim.Adam(model.parameters(), lr=0.001)

    batches_of_labels = list(torch.split(create_dataset.get_labels(), 64))

    print("\nSupervised part!")

    ###################################################################################################

    for epoch in range(10):

        for i in range(len(batches)):
            optimizer_2.zero_grad()
            outputs_supervised = model_2(batches[i])

            #print(outputs_supervised.shape)

            loss = loss_fn_2(outputs_supervised, batches_of_labels[i])

            loss.backward()
            optimizer_2.step()


        sample_tensors = []
        sample_labels = []
        for j in range(31):
            index = random.randint(0, len(batches)-1)
            sample_tensors.append(batches[index])
            sample_labels.append(batches_of_labels[index])


        sample_batch_x = torch.cat(sample_tensors, dim=0)
        sample_batch_y = torch.cat(sample_labels, dim=0)


        plotting_X.plot(sample_batch_x, sample_batch_y, epoch)


        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    
    
    #measuring accuracy


    '''
    test_inputs = create_dataset.get_test_inputs()
    test_labels = create_dataset.get_test_labels()
    
    pred_labels = model_2(test_inputs).argmax(dim=1)
    
    acc = (pred_labels == test_labels).float().mean().item()
    
    print("\nAccuracy = ")
    print(acc * 100)
    
    '''

