import create_dataset
import neural_networks
from torch import optim
import torch.nn as nn


if __name__ == "__main__":
    
    model = neural_networks.AutoEncoder()
    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0009)

    inputs = create_dataset.get_inputs()


    print("\nUnsupervised part!")
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(inputs)
        print(outputs.shape)
        loss = loss_fn(outputs, inputs)
        loss.backward()
        optimizer.step()
        
        
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
         
    
    model_2 = neural_networks.LastLayer(model)  
    
    
    
    model_2.train()
    
    loss_fn_2 = nn.CrossEntropyLoss()
    optimizer_2 = optim.AdamW(model.parameters(), lr=0.007)

    print("\nSupervised part!")
    for epoch in range(10):
        optimizer_2.zero_grad()
        outputs_supervised = model_2(inputs)
        
        loss = loss_fn_2(outputs_supervised, create_dataset.get_labels())
       
        loss.backward(create_graph=True, retain_graph = True)
        optimizer_2.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        
    #output_as_numpy = outputs_supervised.detach().numpy()
    #plt.scatter(output_as_numpy)
    #plt.title("Plot")
    
    
    #measuring accuracy
    
    
    
    test_inputs = create_dataset.get_test_inputs()
    test_labels = create_dataset.get_test_labels()
    
    pred_labels = model_2(test_inputs).argmax(dim=1)
    
    acc = (pred_labels == test_labels).float().mean().item()
    
    print("\nAccuracy = ")
    print(acc * 100)
    