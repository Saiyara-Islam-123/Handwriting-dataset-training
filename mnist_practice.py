import pandas as pd
#this isn't CNN
import torch
from torch import nn
from torch import optim
import numpy as np

#############################################################################################
#loading dataset

df = pd.read_csv('mnist_train.csv')
df_without_labels = df.drop(columns = "label")

#It has columns for pixels. I'll make into one nice matrix. So the resulting df will have 2 columns


matrix = []
row = []
matrices = []


   
for i in range(1000): #row wise. I'm taking a smaller sample
    for column in df_without_labels: #column wise
        column_name = column.split("x")
        row.append(df_without_labels[column][i])
        
        if int(column_name[1]) == 28:
            matrix.append(row)
            row = []
            
    #df_main = df_main.append( {"Label" : df["label"][i], "Matrix" : matrix}, ignore_index=True ) 
    matrices.append(matrix)
    
    row = []
    matrix = []
    
    

#############################################################################################################

#df_unsupervised = df_main.drop(columns = "Label")
#df_supervised = df_main.copy()

matrices_numpy = np.array(matrices)
input_tensor = torch.tensor(matrices_numpy, dtype=torch.float32)


class AutoEncoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            nn.Linear(28 ,25),
            nn.ReLU(),
            nn.Linear(25, 20),
            nn.ReLU()
        )
        
        self.decoder = torch.nn.Sequential(
            
            nn.Linear(20, 25),
            nn.ReLU(),
            nn.Linear(25 ,28),
            nn.ReLU(),
            
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
            

####################################################################################################
class SupervisedNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20, 10), #input of 20. Output of 10
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


####################################################################################################


model = AutoEncoder()
loss_fn = torch.nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

inputs = input_tensor


print("\nUnsupervised part!")
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(inputs)
    print(outputs.shape)
    loss = loss_fn(outputs, inputs)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


#########################################################################################

class LastLayer(nn.Module):
    def __init__(self, pretrained_encoder):
        super(LastLayer, self).__init__()
        self.unsupervised_part = pretrained_encoder.encoder
        
        self.supervised_part = nn.Linear(20, 10)
        
        

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.unsupervised_part(x)
        x = self.supervised_part(x)
        return x
    
    
pretrained_encoder = model
LastLayer(model)


print("\nSupervised part!")
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(inputs)
    print(outputs.shape)
    loss = loss_fn(outputs, inputs)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

