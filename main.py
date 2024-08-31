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
    
    autoencoder = neural_networks.AutoEncoder()
    autoencoder.load_state_dict(torch.load('autoencoder.pth'))

    model_2 = neural_networks.LastLayer(autoencoder)  


    model_2.train()

    print("\nSupervised part!")
    for epoch in range(10):
        optimizer.zero_grad()
        outputs_supervised = model_2(inputs)
        
        loss = loss_fn(outputs_supervised, create_dataset.get_labels())
       
        
        loss.backward(create_graph=True, retain_graph = True)
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        
