import torch
from torch import nn

class AutoEncoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            nn.Linear(28 * 28 ,500),
            nn.ReLU(True),
            
            
            nn.Linear(500, 100),
            nn.ReLU(True),
            
            
            nn.Linear(100, 64),
            nn.ReLU(True),
            
        )
        
        self.decoder = torch.nn.Sequential(
                    
            
            nn.Linear(64, 100),
            nn.ReLU(True),
            
            nn.Linear(100, 500),
            nn.ReLU(True),
        
            
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
        
        self.supervised_part = nn.Sequential(nn.Linear(64, 10))
        
        

    def forward(self, x):
        x = x.view(x.size(0), -1)
        
       
        x = self.autoencoder(x)
        
        
        
        x = self.supervised_part(x)
        
    
        
        return x
