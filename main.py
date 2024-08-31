import torch
import create_dataset
import neural_networks
from torch import optim


if __name__ == "__main__":
    
    model = neural_networks.AutoEncoder()
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
        
        
    torch.save(model.state_dict(), 'autoencoder.pth')

    model_2 = neural_networks.LastLayer(model)

    pretrained_dict = torch.load('autoencoder.pth')
    model_2_dict = model_2.state_dict()
    filtered_dict = {}

    for key in pretrained_dict:
        if key in model_2_dict and "encoder.2" in key:
            filtered_dict[key] = pretrained_dict[key]



    model_2_dict.update(filtered_dict)

    model_2.load_state_dict(model_2_dict)


    model_2.train()

    print("\nSupervised part!")
    for epoch in range(10):
        optimizer.zero_grad()
        outputs_supervised = model_2(inputs)
        
        loss = loss_fn(outputs_supervised, create_dataset.get_labels())
       
        
        loss.backward(create_graph=True, retain_graph = True)
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        
