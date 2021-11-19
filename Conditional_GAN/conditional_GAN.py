# =============================================================================
# Import required libraries 
# =============================================================================
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import timeit

from generator_discriminator import * 
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
plt.imshow(data.data[9], "gray") 

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
def combine_vectors(x, y):
    return torch.cat((x.float(), y.float()), 1)

def get_input_dimensions(noise_dim, data_channels, n_classes):
    generator_input_dim = noise_dim + n_classes
    discriminator_im_chan = data_channels + n_classes
    return generator_input_dim, discriminator_im_chan

PATH_gen = './checkpoints/Generator.pth'
PATH_disc = './checkpoints/Discriminator.pth'
noise_dim = 64
n_classes = 10

gen_input_dim, disc_im_chan = get_input_dimensions(noise_dim, 1, n_classes)
gen = Generator(gen_input_dim)
disc = Discriminator(disc_im_chan)

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
epochs = 200

criterion = nn.BCEWithLogitsLoss()

gen_opt = optim.Adam(gen.parameters(), lr=0.0002)
disc_opt = optim.Adam(disc.parameters(), lr=0.0002)

# =============================================================================
# Training
# =============================================================================
def train(epoch):
    gen.train()
    disc.train()
    
    generator_loss = 0
    discriminator_loss = 0
    
    for batch_idx, (real_images, labels) in enumerate(dataloader):
                
        if train_on_GPU:
            real_images, labels = real_images.cuda(), labels.cuda()

        # get labels one-hot vectors
        one_hot_labels = F.one_hot(labels, n_classes)
        # get images one_hot matrix
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = image_one_hot_labels.repeat(1, 1, real_images.size(2), real_images.size(3))
        #
        
        ## Update discriminator ##
        # zero the gradients parameter
        disc_opt.zero_grad()
        # generating fake images
        noise = get_noise(real_images.size(0), noise_dim)
        fake_image = gen(combine_vectors(noise, one_hot_labels))
        #
        disc_fake_pred = disc(combine_vectors(fake_image.detach(), image_one_hot_labels))
        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        
        disc_real_pred = disc(combine_vectors(real_images, image_one_hot_labels))
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        #
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        # backward pass: compute gradient of the loss with respect to model parameters
        disc_loss.backward(retain_graph=True)
        # parameters update
        disc_opt.step()
        
        ## Update generator ##
        # zero the gradients parameter
        gen_opt.zero_grad()
        
        disc_fake_pred = disc(combine_vectors(fake_image, image_one_hot_labels))
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
def number_generator(number):
    if train_on_GPU:
        one_hot_label = F.one_hot(torch.Tensor([number]).long(), n_classes).cuda()
    else:
        one_hot_label = F.one_hot(torch.Tensor([number]).long(), n_classes)
    noise = get_noise(1, noise_dim)
    gen.eval()
    with torch.no_grad():
        fake_image = gen(combine_vectors(noise, one_hot_label))
    img = (fake_image / 2) + 0.5
    img = img.detach().cpu()
    img = img.squeeze(0).permute(1, 2, 0)
    plt.imshow(img, "gray")
number_generator(5)