import torch
from torch import nn


class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(

            nn.Conv2d(in_channels= 1, out_channels= 1, kernel_size=4, stride=2, padding=1),  # 1 input channel (grayscale), 16 output channels
            nn.ReLU(True),
            nn.Conv2d(in_channels= 1, out_channels= 1,kernel_size=2, stride=2, padding=1),  # 32 output channels
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64, 64),  # Linear layer after convs, 256 units
            nn.ReLU(True)




        )



        self.decoder = torch.nn.Sequential(

            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Unflatten(1, (1, 8, 8)),
            nn.ConvTranspose2d(in_channels= 1, out_channels= 1, kernel_size=2, stride=2, padding=1),
            # Correcting stride/padding
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels= 1, out_channels= 1, kernel_size=4, stride=2, padding=1),  # Exact match for 28x28
            nn.Sigmoid()  # Sigmoid to ensure the output is between 0 and 1

        )


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


################################################################################################

class LastLayer(nn.Module):
    def __init__(self, autoencoder):
        super(LastLayer, self).__init__()

        self.autoencoder_output = None
        self.autoencoder = autoencoder.encoder

        self.supervised_part = nn.Sequential(nn.Linear(64, 10))

    def forward(self, x):
        x = self.autoencoder(x)

        self.autoencoder_output = x

        x = x.view(x.size(0), -1)

        x = self.supervised_part(x)

        return x