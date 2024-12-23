import torch
from torch import nn


class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoded = None

        self.encoder = torch.nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 1 input channel (grayscale), 16 output channels
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 32 output channels
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64 output channels
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),  # Linear layer after convs, 256 units
            nn.ReLU(True)




        )



        self.decoder = torch.nn.Sequential(

            nn.Linear(128, 64 * 4 * 4),
            nn.ReLU(True),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            # Correcting stride/padding
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # Exact match for 28x28
            nn.Sigmoid()  # Sigmoid to ensure the output is between 0 and 1

        )


    def forward(self, x):
        self.encoded = self.encoder(x)
        decoded = self.decoder(self.encoded)
        return decoded


################################################################################################

class LastLayer(nn.Module):
    def __init__(self, autoencoder):
        super(LastLayer, self).__init__()

        self.autoencoder_output = None
        self.autoencoder = autoencoder.encoder

        self.supervised_part = nn.Sequential(nn.Linear(128, 10))

    def forward(self, x):
        x = self.autoencoder(x)

        self.autoencoder_output = x

        x = x.view(x.size(0), -1)

        x = self.supervised_part(x)

        return x