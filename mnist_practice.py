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
labels = []

   
for i in range(100): #row wise. I'm taking a smaller sample
    for column in df_without_labels: #column wise
        column_name = column.split("x")
        row.append(df_without_labels[column][i])
        
        if int(column_name[1]) == 28:
            matrix.append(row)
            row = []
            
    #df_main = df_main.append( {"Label" : df["label"][i], "Matrix" : matrix}, ignore_index=True ) 
    matrices.append(matrix)
    labels.append(df["label"][i])
    
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
    
    
torch.save(model.state_dict(), 'autoencoder.pth')


#########################################################################################

class LastLayer(nn.Module):
    def __init__(self):
        super(LastLayer, self).__init__()
        
        self.supervised_part = nn.Linear(784, 1)
        
        

    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        print("\nThis is x's size.")
        print(x.shape)
        
        x = self.supervised_part(x)
        return x
    
    
model_2 = LastLayer()

pretrained_dict = torch.load('autoencoder.pth')
model_2_dict = model_2.state_dict()
filtered_dict = {}

for key in pretrained_dict:
    if key in model_2_dict and "encoder.2" in key:
        filtered_dict[key] = pretrained_dict[key]



model_2_dict.update(filtered_dict)

model_2.load_state_dict(model_2_dict)


def custom_loss(output, target):
    loss = (output - target).pow(2).mean()
    return loss


print("\nSupervised part!")
for epoch in range(10):
    optimizer.zero_grad()
    outputs_supervised = model_2(outputs)
    print("\nThis is supervised output")
    print(outputs_supervised.shape)
    
    print(torch.tensor(labels).shape)
    
    loss = custom_loss(outputs_supervised, torch.tensor(labels))
   
    
    loss.backward(retain_graph=True, create_graph=True)
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

