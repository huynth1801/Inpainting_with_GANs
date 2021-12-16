import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        """[summary]

        Args:
            nz (int): length of latent vector (i.e. input size of generator)
            ngf (int): size of feature map in generator
            nc (int): number of channels in trainining images
        """
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.main = nn.Sequential(
                    # Block 1
                    nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf*8),
                    nn.ReLU(True),
                    # Block 2: state size (ngf*8)x4x4
                    nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf*4),
                    nn.ReLU(True),
                    # Block 3: state size (ngf*4)x8x8
                    nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf*2),
                    nn.ReLU(True),
                    # Block 4: state size (ngf*2)x16x16
                    nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf),
                    nn.ReLU(True),
                    # Block 5: state size (ngf)x32x32
                    nn.ConvTranspose2d(ngf, ngf//2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf//2),
                    nn.ReLU(True),
                    # Block 6: state size (ngf//2)x64x64
                    nn.ConvTranspose2d(ngf//2, nc, 4, 2, 1, bias=False),
                    nn.Tanh()
                    # state size. (nc) x 128 x 128
        )

    def forward(self, x):
        # [batch, nz, 1, 1]
        return self.main(x)

if __name__ == "__main__":
    pass