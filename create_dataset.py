#loading dataset
import numpy as np
import pandas as pd
import torch


df = pd.read_csv('mnist_train.csv')
df_without_labels = df.drop(columns = "label")

#It has columns for pixels. I'll make into one nice matrix. So the resulting df will have 2 columns


matrix = []
matrices = []
labels = []

   
for i in range(10000): #row wise. I'm taking a smaller sample
    for column in df_without_labels: #column wise
        column_name = column.split("x")
        matrix.append(df_without_labels[column][i])
        
            
    #df_main = df_main.append( {"Label" : df["label"][i], "Matrix" : matrix}, ignore_index=True ) 
    matrices.append(matrix)
    labels.append(df["label"][i])
    

    matrix = []
    
    

#############################################################################################################

#df_unsupervised = df_main.drop(columns = "Label")
#df_supervised = df_main.copy()



matrices_numpy = np.array(matrices)
input_tensor = torch.tensor(matrices_numpy, dtype=torch.float32)

def get_inputs():
    return input_tensor

def get_labels():
    return torch.tensor(labels)

