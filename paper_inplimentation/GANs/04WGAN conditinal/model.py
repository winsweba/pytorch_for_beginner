""" 
Discriminator and Generator implementation from WGAN paper
"""

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def _init__(self, channels_img,features_d, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.img_size =img_size
        self.disc = nn.Sequential(
            nn.Conv2d(
                channels_img + 1, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1),
            self._block(features_d*2, features_d*4, 4, 2, 1),
            self._block(features_d*4, features_d*8, 4, 2, 1),
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=1),
        )
        self.embed = nn.Embedding(num_classes, img_size*img_size)
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            # nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels , affine=True), #LayerNorm == InstanceNorm
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x =torch.cat([x, embedding], dim=1) # N x C x img_size (H) x img_size (w)
        return self.disc(x)
    

class Generator(nn.Module):
    def __init__(self, z_dim, channel_img, featured_g,  num_classes, img_size, embed_ize):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.gen = nn.Sequential(
            self._block(z_dim + embed_ize, featured_g*16, 4, 1, 0), # N x f_g*16 x 4 x4
            self._block(featured_g*16, featured_g*8, 4, 2, 1), # 8x8
            self._block(featured_g*8, featured_g*4, 4, 2, 1),# 16x16
            self._block(featured_g*4, featured_g*2, 4, 2, 1), #32x32
            nn.ConvTranspose2d(
                featured_g*2, channel_img, kernel_size=4, stride=2, padding=1,
            ),
            nn.Tanh(), #[-1, 1]
        )
        self.embed = nn.Embedding(num_classes, embed_ize)

    def _block(self, in_channels, out_channels, kernel_size, string, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3) 
        x = torch.cat([x, embedding], dim=1)
        return self.gen(x)
    
def initializer_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    initializer_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)
    gen = Generator(z_dim, in_channels, 8)
    initializer_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape ==  (N, in_channels, H, W)
    print("Success")


test()