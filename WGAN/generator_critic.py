import torch
import torch.nn as nn
    
class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim=64, image_channel=3):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # 1 * 1 * 100
            nn.ConvTranspose2d(in_channels=noise_dim, out_channels=hidden_dim * 8, kernel_size=4, stride=1),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(inplace=True),
            # 4 * 4 * 512
            nn.ConvTranspose2d(in_channels=hidden_dim * 8, out_channels=hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            # 8 * 8 * 256
            nn.ConvTranspose2d(in_channels=hidden_dim * 4, out_channels=hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            # 16 * 16 * 128
            nn.ConvTranspose2d(in_channels=hidden_dim * 2, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # 32 * 32 * 64
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=image_channel, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            # 64 * 64 * 3
        )
        
    def forward(self, noise):
        noise = noise.view(noise.size(0), noise.size(1), 1, 1) # (batch_size, noise_dim, height=1, width=1)
        return self.gen(noise)
    
    
class Critic(nn.Module):
    def __init__(self, hidden_dim=64, image_channel=3):
        super(Critic, self).__init__()
        self.disc = nn.Sequential(
            # 64 * 64 * 3
            nn.Conv2d(in_channels=image_channel, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 32 * 32 * 64
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 16 * 16 * 128
            nn.Conv2d(in_channels=hidden_dim * 2, out_channels=hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 8 * 8 * 256
            nn.Conv2d(in_channels=hidden_dim * 4, out_channels=hidden_dim * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 4 * 4 * 512
            nn.Conv2d(in_channels=hidden_dim * 8, out_channels=1, kernel_size=4, stride=1),
            # 1 * 1 * 1

        )
        
    def forward(self, image):
        image = self.disc(image)
        image = torch.flatten(image, 1)
        return image