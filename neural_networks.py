import torch
from torch import nn

class AutoEncoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            nn.Linear(28 * 28 ,500),
            nn.ReLU(True),
            
            nn.Dropout(0.5),
            
            nn.Linear(500, 300),
            nn.ReLU(True),
            
            nn.Linear(300, 100),
            nn.ReLU(True),
            
            
            nn.Linear(100, 50),
            nn.ReLU(True),
            
            
            nn.Linear(50, 20),
            nn.ReLU(True),
            
        )
        
        self.decoder = torch.nn.Sequential(
            
            nn.Linear(20, 50),
            nn.Sigmoid(),
            
            nn.Linear(50, 100),
            nn.Sigmoid(),
            
            
            nn.Linear(100, 300),
            nn.Sigmoid(),
            
            nn.Linear(300, 500),
            nn.Sigmoid(),
        
            
            nn.Linear(500 ,28 * 28),
            nn.Sigmoid(),
            
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
################################################################################################    
    
class LastLayer(nn.Module):
    def __init__(self, autoencoder):
        super(LastLayer, self).__init__()
        
        self.autoencoder = autoencoder.encoder
        
        self.supervised_part = nn.Sequential(nn.Linear(20, 10), nn.Sigmoid())
        
        

    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        print("\nThis is x's size.")
        print(x.shape)
        x = self.autoencoder(x)
        
        print(x.shape)
        
        x = self.supervised_part(x)
        
        print(x.shape)
        
        return x
