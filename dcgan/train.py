import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import random
import statistics
import os
import sys
import argparse
import torch.utils.tensorboard as tb

from dcgan import weights_init, Generator, Discriminator

from functools import partial
from tqdm.auto import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

sys.path.append('../')
import mydataset


def main():
    # expID
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./project/proj0')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--data_num', type=int, default=12000)
    parser.add_argument('--split_seed', type=int, default=1)
    args = parser.parse_args()
    os.makedirs(args.path)

    tb_path = os.path.join(args.path, "tensorboard")
    os.makedirs(tb_path, exist_ok=True)
    tb_logger = tb.SummaryWriter(log_dir=tb_path)


    # Set random seed for reproducibility.
    seed = random.randint(1000, 2000)
    random.seed(seed)
    torch.manual_seed(seed)
    print("Random Seed: ", seed)

    # Parameters to define the model.
    params = {
        "bsize" : 128,# Batch size during training.
        'imsize' : 64,# Spatial size of training images. All images will be resized to this size during preprocessing.
        'nc' : 3,# Number of channles in the training images. For coloured images this is 3.
        'nz' : 100,# Size of the Z latent vector (the input to the generator).
        'ngf' : 64,# Size of feature maps in the generator. The depth will be multiples of this.
        'ndf' : 64,# Size of features maps in the discriminator. The depth will be multiples of this.
        'nepochs' : 600,# Number of training epochs.
        'lr' : 0.0002,# Learning rate for optimizers
        'beta1' : 0.5,# Beta1 hyperparam for Adam optimizer
        'save_epoch' : 100}# Save step.

    # Use GPU is available else use CPU.
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    print(device, " will be used.\n")

    # Get the data.
    if args.dataset == 'cifar10':
        params['imsize'] = 32
        params['ngf'] = 32
        params['ndf'] = 32
        dataloader, testloader, info = mydataset.get_dataloader_cifar10(params['imsize'], params['bsize'], args.data_num, args.split_seed, normalize=True)
    elif args.dataset == 'celeba':
        dataloader, testloader, info = mydataset.get_dataloader_celeba(params['imsize'], params['bsize'], args.data_num, args.split_seed, normalize=True)

    with open(os.path.join(args.path, 'info.csv'), 'a') as f:
            print(f"{args.dataset}, {info[0]}, {info[1]}", file=f)

    # Create the generator.
    netG = Generator(params).to(device)
    # Apply the weights_init() function to randomly initialize all
    # weights to mean=0.0, stddev=0.2
    netG.apply(weights_init)
    # Print the model.
    print(netG)

    # Create the discriminator.
    netD = Discriminator(params).to(device)
    # Apply the weights_init() function to randomly initialize all
    # weights to mean=0.0, stddev=0.2
    netD.apply(weights_init)
    # Print the model.
    print(netD)

    # Binary Cross Entropy loss function.
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)

    real_label = 1
    fake_label = 0

    # Optimizer for the discriminator.
    optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
    # Optimizer for the generator.
    optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))

    print("Starting Training Loop...")
    print("-"*25)

    for epoch in tqdm(range(params['nepochs'])):
        G_loss = []
        D_loss = []
        tr_loss = []
        for i, (data, cl) in enumerate(dataloader):
            # Transfer data tensor to GPU/CPU (device)
            real_data = data.to(device)
            # Get batch size. Can be different from params['nbsize'] for last batch in epoch.
            b_size = real_data.size(0)
            
            # Make accumalated gradients of the discriminator zero.
            netD.zero_grad()
            # Create labels for the real data. (label=1)
            label = torch.full((b_size, ), real_label, dtype=torch.float32, device=device)
            output = netD(real_data).view(-1)
            errD_real = criterion(output, label)

            # Calculate gradients for backpropagation.
            errD_real.backward()
            
            # Sample random data from a unit normal distribution.
            noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
            # Generate fake data (images).
            fake_data = netG(noise)
            # Create labels for fake data. (label=0)
            label.fill_(fake_label)
            # Calculate the output of the discriminator of the fake data.
            # As no gradients w.r.t. the generator parameters are to be
            # calculated, detach() is used. Hence, only gradients w.r.t. the
            # discriminator parameters will be calculated.
            # This is done because the loss functions for the discriminator
            # and the generator are slightly different.
            output = netD(fake_data.detach()).view(-1)
            errD_fake = criterion(output, label)
            # Calculate gradients for backpropagation.
            errD_fake.backward()

            # Net discriminator loss.
            errD = errD_real + errD_fake
            # Update discriminator parameters.
            optimizerD.step()
            
            # Make accumalted gradients of the generator zero.
            netG.zero_grad()
            # We want the fake data to be classified as real. Hence
            # real_label are used. (label=1)
            label.fill_(real_label)
            # No detach() is used here as we want to calculate the gradients w.r.t.
            # the generator this time.
            output = netD(fake_data).view(-1)
            errG = criterion(output, label)
            # Gradients for backpropagation are calculated.
            # Gradients w.r.t. both the generator and the discriminator
            # parameters are calculated, however, the generator's optimizer
            # will only update the parameters of the generator. The discriminator
            # gradients will be set to zero in the next iteration by netD.zero_grad()
            errG.backward()

            # Update generator parameters.
            optimizerG.step()

            # Save the losses for plotting.
            G_loss.append(errG.item())
            D_loss.append(errD.item())
            tr_loss.append(errD_real.item())

            # Check how the generator is doing by saving G's output on a fixed noise.
            if (epoch % 100 == 0) and (i == len(dataloader)-1):
                with torch.no_grad():
                    fake_data = netG(fixed_noise).detach().cpu()
                result = np.transpose(vutils.make_grid(fake_data, padding=2, normalize=True), (1,2,0))
                plt.imsave(os.path.join(args.path, f"sample_{epoch}.png"), result.to('cpu').detach().numpy().copy())

        
        G_loss_mean = statistics.mean(G_loss)
        D_loss_mean = statistics.mean(D_loss)
        tr_loss_mean = statistics.mean(tr_loss)

        netD.eval()
        val_loss = []
        val_skip = 3
        rand = random.randint(0, val_skip-1)
        with torch.no_grad():
            for i, (data, cl) in enumerate(testloader):
                if i % val_skip != rand:
                    continue
                val_data = data.to(device)
                b_size = val_data.size(0)
                label = torch.full((b_size, ), real_label, dtype=torch.float32, device=device)
                output = netD(val_data).view(-1)
                errD_val = criterion(output, label)
                val_loss.append(errD_val.item())
        val_loss_mean = statistics.mean(val_loss)
        netD.train()

        tb_logger.add_scalars('gd_loss', {'generator_loss':G_loss_mean, 'discriminator_loss':D_loss_mean}, epoch)
        tb_logger.add_scalars('tv_loss', {'train_loss':tr_loss_mean, 'validation_loss':val_loss_mean}, epoch)

        # Save the model.
        if epoch % params['save_epoch'] == 0:
            torch.save({
                'generator' : netG.state_dict(),
                'discriminator' : netD.state_dict(),
                'optimizerG' : optimizerG.state_dict(),
                'optimizerD' : optimizerD.state_dict(),
                'params' : params
                }, os.path.join(args.path, f'model_epoch_{epoch}.pth'))

    # Save the final trained model.
    torch.save({
        'generator' : netG.state_dict(),
        'discriminator' : netD.state_dict(),
        'optimizerG' : optimizerG.state_dict(),
        'optimizerD' : optimizerD.state_dict(),
        'params' : params
        }, os.path.join(args.path, f"model_epoch_{params['nepochs']}.pth"))

    tb_logger.close()


if __name__ == '__main__':
    main()
