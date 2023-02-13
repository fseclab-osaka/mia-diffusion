# Membership Inference Attacks against Diffusion Models

## Requirements
See `myenv.yml`.

## Dataset
CIFAR-10: Auto download to `dataset/cifar10`.   
CelebA: Manual download to `dataset/celeba`.

## Model
### DDIM
Training & Sampling
```bash
cd ddim
python main.py --config <cifar10/celeba>.yml --exp "PROJECT_PATH" --ni
python main.py --config <cifar10/celeba>.yml --exp "PROJECT_PATH" --sample --fid --ni
```

### DCGAN
Training & Sampling
```bash
cd dcgan
python train.py --path "PROJECT_PATH" --dataset <cifar10/celeba>
python generate.py --project_dir "PROJECT_PATH"
```
FID Evaluation
```bash
cd TTUR
python fid.py ../<ddim/dcgan>/"PROJECT_PATH"/samples_"EPOCH_NUM"/images ./<fid_stats_cifar10_train/fid_stats_celeba_60k>.npz
```
## Attack
### White-box
```bash
cd attacks
python <fa_ddim/fa_dcgan>.py
```

### Black-box
```bash
cd attacks
python fbb.py
```

## Acknowledgements
Our implementation uses the source code from the following repositories:

<https://github.com/DingfanChen/GAN-Leaks>   
<https://github.com/ermongroup/ddim>   
<https://github.com/Natsu6767/DCGAN-PyTorch>   
<https://github.com/bioinf-jku/TTUR> (tensorflow)   