import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        """[summary]

        Args:
            nc ([int]): [number of channels in trainining images]
            ndf ([int]): [size of feature map in discriminator]
        """
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.main = nn.Sequential(
                    # Block 1: (nc)x128x128
                    nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    # Block 2: state size (ndf)x64x64
                    nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    # Block 3: state size (ndf)x32x32
                    nn.Conv2d(ndf, ndf*2 , 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf*2),
                    nn.LeakyReLU(0.2, inplace=True),
                    # Block 4: state size (ndf*2)x16x16
                    nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf*4),
                    nn.LeakyReLU(0.2, inplace=True),
                    # Block 5: state size (ndf*4)x8x8
                    nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf*8),
                    nn.LeakyReLU(0.2, inplace=True),
                    # Block 6: state size. (ndf*8)x4x4
                    nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                    nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

if __name__ == "__main__":
    pass