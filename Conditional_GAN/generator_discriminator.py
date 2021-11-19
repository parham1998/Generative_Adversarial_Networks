import torch
import torch.nn as nn
    
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, image_channel=1):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # 1 * 1 * (64+10)
            nn.ConvTranspose2d(in_channels=input_dim, out_channels=hidden_dim * 4, kernel_size=3, stride=2),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            # 3 * 3 * 256
            nn.ConvTranspose2d(in_channels=hidden_dim * 4, out_channels=hidden_dim * 2, kernel_size=4, stride=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            # 6 * 6 * 128
            nn.ConvTranspose2d(in_channels=hidden_dim * 2, out_channels=hidden_dim, kernel_size=3, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # 13 * 13 * 64
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=image_channel, kernel_size=4, stride=2),
            nn.Tanh(),
            # 28 * 28 * 1
        )
        
    def forward(self, class_noise):
        class_noise = class_noise.view(class_noise.size(0), class_noise.size(1), 1, 1) # (batch_size, channels, height, width)
        return self.gen(class_noise)
    
    
class Discriminator(nn.Module):
    def __init__(self, image_channel, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # 28 * 28 * (1+10)
            nn.Conv2d(in_channels=image_channel, out_channels=hidden_dim, kernel_size=4, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 13 * 13 * 64
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=4, stride=2),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 5 * 5 * 128
            nn.Conv2d(in_channels=hidden_dim * 2, out_channels=1, kernel_size=4, stride=2),
            # 1 * 1 * 1
        )
        
    def forward(self, image):
        image = self.disc(image)
        image = torch.flatten(image, 1)
        return image