import torch
from torch import nn

class AutoEncoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            nn.Linear(28 * 28 ,25),
            nn.ReLU(),
            nn.Linear(25, 20),
            nn.ReLU()
        )
        
        self.decoder = torch.nn.Sequential(
            
            nn.Linear(20, 25),
            nn.ReLU(),
            nn.Linear(25 ,28 * 28),
            nn.ReLU(),
            
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
        
        self.supervised_part = nn.Sequential(nn.Linear(20, 10), nn.ReLU())
        
        

    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        print("\nThis is x's size.")
        print(x.shape)
        x = self.autoencoder(x)
        
        print(x.shape)
        
        x = self.supervised_part(x)
        
        print(x.shape)
        
        return x
