# =============================================================================
# Import required libraries 
# =============================================================================
import os
from PIL import Image
from sklearn.utils import shuffle
from tqdm.auto import tqdm

import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch import nn, optim

import pandas as pd
import matplotlib.pyplot as plt
import timeit

from generator_critic import * 
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
batch_size = 30

class DogDataset(torch.utils.data.Dataset):                                 
    def __init__(self, root, csv_file, transforms=None):                       
        self.root = root
        self.csv_file = shuffle(pd.read_csv(csv_file), random_state=20)
        self.transforms = transforms  
        self.classes = {"bloodhound": 0, 
                        "Saluki": 1, 
                        "golden_retriever": 2, 
                        "Old_English_sheepdog": 3, 
                        "German_shepherd": 4,
                        "Doberman": 5, 
                        "Great_Dane": 6, 
                        "Siberian_husky": 7, 
                        "chow": 8, 
                        "toy_poodle": 9}  
            
    def __getitem__(self, idx):     
        label = self.csv_file.iloc[idx, 1]
        label = self.classes[label]
        img_name = os.path.join(self.root, self.csv_file.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        return image, torch.Tensor([label]).long()                                                     
    
    def __len__(self):                                                          
        return len(self.csv_file) 

transform = transforms.Compose([
                     transforms.Resize((64, 64)),
                     transforms.ToTensor(),
                     transforms.Normalize(
                         mean=(0.5, 0.5, 0.5), 
                         std=(0.5, 0.5, 0.5)
                     ),
                 ])

data = DogDataset(root='./data/images',
                  csv_file='./data/file.csv', 
                  transforms=transform)

# show one image
def imshow(tensor):
    img = (tensor / 2) + 0.5
    img = img.permute(1, 2, 0)
    plt.imshow(img)
img, label = data[5]
imshow(img)

dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

# one batch of data
# images: (batch_size, channels, height, width)
images, labels = iter(dataloader).next()

def show_tensor_images(image_tensor, num_images=9, size=(3, 64, 64)):
    image_tensor = (image_tensor / 2) + 0.5
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=3)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
show_tensor_images(images)

# =============================================================================
# Deep convolutional GAN
# =============================================================================   
PATH_gen = './checkpoints/Generator.pth'
PATH_disc = './checkpoints/Critic.pth'
noise_dim = 100
gen = Generator(noise_dim)
crit = Critic()

if train_on_GPU:
    gen.cuda()
    crit.cuda()
    print('\n net can be trained on gpu')
    
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
gen = gen.apply(weights_init)
crit = crit.apply(weights_init)

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
    crit.load_state_dict(torch.load(PATH_disc))
    crit.cuda()
    print('\n net can be trained on gpu') 
else:
    gen.load_state_dict(torch.load(PATH_gen, torch.device('cpu')))
    crit.load_state_dict(torch.load(PATH_disc, torch.device('cpu')))

# =============================================================================
# Gradient penalty
# =============================================================================
def get_gradient(crit, real, fake, epsilon):
    mixed_images = epsilon * real + (1 - epsilon) * fake

    mixed_scores = crit(mixed_images)
    
    # take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient

def gradient_penalty(gradient):
    gradient = torch.flatten(gradient, 1)
    # calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)
    # penalize the mean squared distance of the gradient norms from 1
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty

# =============================================================================
# Specify loss function and optimizer
# =============================================================================
epochs = 200

c_lambda = 10
crit_repeats = 5

gen_opt = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
crit_opt = optim.Adam(crit.parameters(), lr=0.0002, betas=(0.5, 0.999))

# =============================================================================
# Training
# =============================================================================
def train(epoch):
    gen.train()
    crit.train()
    
    generator_loss = 0
    critic_loss = 0
    
    for batch_idx, (real_images, labels) in tqdm(enumerate(dataloader)):
                
        if train_on_GPU:
            real_images = real_images.cuda()

        for _ in range(crit_repeats):
            ### Update critic ###
            crit_opt.zero_grad()

            fake_noise = get_noise(real_images.size(0), noise_dim)
            fake_image = gen(fake_noise)
            #
            crit_fake_pred = crit(fake_image.detach())
            crit_real_pred = crit(real_images)
            #
            epsilon = torch.rand(real_images.size(0), 1, 1, 1, requires_grad=True).cuda()
            gradient = get_gradient(crit, real_images, fake_image.detach(), epsilon)
            gp = gradient_penalty(gradient)
            #
            crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp            
            # Update gradients
            crit_loss.backward(retain_graph=True)
            # Update optimizer
            crit_opt.step()

            critic_loss += crit_loss.item() / crit_repeats

        ### Update generator ###
        gen_opt.zero_grad()

        fake_noise = get_noise(real_images.size(0), noise_dim)
        fake_2 = gen(fake_noise)

        crit_fake_pred = crit(fake_2)
        #
        gen_loss = -1 * torch.mean(crit_fake_pred)
        #
        gen_loss.backward()
        # Update the weights
        gen_opt.step()

        generator_loss += gen_loss.item()

    print('Epoch: {} \t Critic Loss: {:.3f} \t Generator Loss: {:.3f}'.format(epoch+1, critic_loss/(batch_idx+1), generator_loss/(batch_idx+1)))
    # save model
    torch.save(gen.state_dict(), PATH_gen)
    torch.save(crit.state_dict(), PATH_disc)
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
    plt.imshow(img)
number_generate()