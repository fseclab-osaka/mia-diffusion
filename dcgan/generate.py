import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.utils as vutils

from functools import partial
from tqdm.auto import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from dcgan import Generator


def main(epoch):
	parser = argparse.ArgumentParser()
	parser.add_argument('--project_dir', default='./project/proj0/')
	parser.add_argument('--num_output', default=16, help='Number of generated outputs')
	args = parser.parse_args()

	print(args.num_output)

	load_path = f"{args.project_dir}model_epoch_{epoch}.pth"
	sample_path = f"{args.project_dir}samples_{epoch}"
	os.makedirs(sample_path, exist_ok=True)


	# Load the checkpoint file.
	state_dict = torch.load(load_path)

	# Set the device to run on: GPU or CPU.
	device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
	# Get the 'params' dictionary from the loaded state_dict.
	params = state_dict['params']

	# Create the generator network.
	netG = Generator(params).to(device)
	# Load the trained generator weights.
	netG.load_state_dict(state_dict['generator'])
	print(netG)

	# Get latent vector Z from unit normal distribution.
	noise = torch.randn(int(args.num_output), params['nz'], 1, 1, device=device)

	# Turn off gradient calculation to speed up the process.
	with torch.no_grad():
		# Get generated image from the noise vector using
		# the trained generator.
		generated_img = netG(noise).detach().cpu()

	print("noise:" + str(noise.shape))
	print("generated_img:" + str(generated_img.shape))

	img_r01 = (generated_img + 1.) / 2.
	noise_np = noise.to('cpu').detach().numpy().copy()
	img_r01_np = img_r01.to('cpu').detach().numpy().copy()
	#np.savez_compressed(os.path.join(sample_path, 'generated.npz'), noise=noise_np, img_r01=img_r01_np)

	png_path = os.path.join(sample_path, "images")
	os.makedirs(png_path, exist_ok=True)
	for img_id in tqdm(range(int(args.num_output))):
		vutils.save_image(img_r01[img_id], os.path.join(png_path, f"{img_id}.png"))

	# Display the generated image.
	plt.axis("off")
	plt.title("Generated Images")
	result = np.transpose(vutils.make_grid(img_r01[:16], nrow=4, padding=1, normalize=False), (1,2,0))
	print("result:" + str(result.shape))
	plt.imsave(os.path.join(sample_path, "sample.png"), result.to('cpu').detach().numpy().copy())


if __name__ == '__main__':
	epochs = [6000]
	for epoch in epochs:
		print(f"epoch:{epoch}")
		main(epoch)
