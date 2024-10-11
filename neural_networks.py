
import torch
from torch import nn

class AutoEncoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(

            nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),

            nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            
        )
        
        self.decoder = torch.nn.Sequential(

            nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),

            nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0),
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
        
        self.supervised_part = nn.Sequential(nn.Linear(49, 10))
        
        

    def forward(self, x):

        x = self.autoencoder(x)

        x = x.view(x.size(0), -1)

        x = self.supervised_part(x)
        
    
        
        return x
