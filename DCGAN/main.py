# =============================================================================
# Import required libraries 
# =============================================================================
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch import nn, optim

import matplotlib.pyplot as plt
import timeit

from deep_convolutional_GAN import * 
torch.manual_seed(0) 

# =============================================================================
# Check if CUDA is available
# =============================================================================
train_on_GPU = torch.cuda.is_available()
if not train_on_GPU:
    print('CUDA is not available. Training on CPU ...')
else:
    print('CUDA is available! Training on GPU ...') 
    print(torch.cuda.get_device_properties('cuda'))
    
# =============================================================================
# Load data & data preprocessing
# =============================================================================
batch_size = 128

transform = transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(
                         mean=(0.5), 
                         std=(0.5)
                     ),
                 ])

data = torchvision.datasets.MNIST(
    root="./data", 
    download=True, 
    transform=transform)

# show one image
plt.imshow(data.data[9]) 

dataloader = DataLoader(
    data, 
    batch_size=batch_size,
    shuffle=True)

# one batch of data
# images: (batch_size, channels, height, width)
images, labels = iter(dataloader).next()

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_tensor = (image_tensor / 2) + 0.5
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
show_tensor_images(images)

# =============================================================================
# Deep convolutional GAN
# =============================================================================   
PATH_gen = './checkpoints/Generator.pth'
PATH_disc = './checkpoints/Discriminator.pth'
noise_dim = 64
gen = Generator(noise_dim)
disc = Discriminator()

if train_on_GPU:
    gen.cuda()
    disc.cuda()
    print('\n net can be trained on gpu')
    
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

def get_noise(n_samples, noise_dim):
    if train_on_GPU:
        return torch.randn(n_samples, noise_dim).cuda()
    else:
        return torch.randn(n_samples, noise_dim)

# =============================================================================
# Load model
# =============================================================================
if train_on_GPU:
    gen.load_state_dict(torch.load(PATH_gen))
    gen.cuda()
    disc.load_state_dict(torch.load(PATH_disc))
    disc.cuda()
    print('\n net can be trained on gpu') 
else:
    gen.load_state_dict(torch.load(PATH_gen, torch.device('cpu')))
    disc.load_state_dict(torch.load(PATH_disc, torch.device('cpu')))

# =============================================================================
# Specify loss function and optimizer
# =============================================================================
epochs = 50

criterion = nn.BCEWithLogitsLoss()

gen_opt = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
disc_opt = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))

# =============================================================================
# Training
# =============================================================================
def train(epoch):
    gen.train()
    disc.train()
    
    generator_loss = 0
    discriminator_loss = 0
    
    for batch_idx, (real_images, _) in enumerate(dataloader):
                
        if train_on_GPU:
            real_images = real_images.cuda()

        ## Update discriminator ##
        # zero the gradients parameter
        disc_opt.zero_grad()
        # generating fake images
        noise = get_noise(real_images.size(0), noise_dim)
        fake_image = gen(noise)
        #
        disc_fake_pred = disc(fake_image.detach())
        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_pred = disc(real_images)
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        # backward pass: compute gradient of the loss with respect to model parameters
        disc_loss.backward(retain_graph=True)
        # parameters update
        disc_opt.step()
        
        ## Update generator ##
        # zero the gradients parameter
        gen_opt.zero_grad()
        # generating fake images
        noise = get_noise(real_images.size(0), noise_dim)
        fake_image2 = gen(noise)
        #
        disc_fake_pred = disc(fake_image2)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        gen_loss.backward()
        gen_opt.step()

        discriminator_loss += disc_loss.item() 
        generator_loss += gen_loss.item()

    print('Epoch: {} \t Discriminator Loss: {:.3f} \t Generator Loss: {:.3f}'.format(epoch+1, discriminator_loss/(batch_idx+1), generator_loss/(batch_idx+1)))
    # save model
    torch.save(gen.state_dict(), PATH_gen)
    torch.save(disc.state_dict(), PATH_disc)
    # show real and fake images
    show_tensor_images(fake_image)
    show_tensor_images(real_images)
    
print('==> Start Training ...')
for epoch in range(epochs):
    start = timeit.default_timer()
    train(epoch)
    stop = timeit.default_timer()
    print('time: {:.3f}'.format(stop - start))
print('==> End of training ...')

# =============================================================================
# Test on random noise
# =============================================================================
def number_generate():
    noise = get_noise(1, noise_dim)
    gen.eval()
    with torch.no_grad():
        fake_image = gen(noise)
    img = (fake_image / 2) + 0.5
    img = img.detach().cpu()
    img = img.squeeze(0).permute(1, 2, 0)
    plt.imshow(img, "gray")
number_generate()