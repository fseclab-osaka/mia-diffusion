import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import glob
import shutil
import random
import torch
import yaml
import pickle
import csv

from functools import partial
from tqdm.auto import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from tools.utils import *
import tools.eval_roc as eval_roc

sys.path.append('../ddim/')
from models.diffusion import Model
from models.ema import EMAHelper
from functions.losses import loss_registry
from datasets import data_transform
from runners.diffusion import get_beta_schedule
from main import dict2namespace

sys.path.append('../')
import mydataset


def main(ckpt):
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', default="~/mia_diffusion/ddim/project/proj0/")
    parser.add_argument('--data_skip', type=int, default=1)
    args = parser.parse_args()

    model_path = f"{args.project_dir}logs/ddim/ckpt_{ckpt}.pth"
    save_dir = f"{args.project_dir}samples_{ckpt}/fa_result"
    data_skip = args.data_skip
    
    #time_range = range(0, 1000, 25)
    time_range = [350]

    os.makedirs(save_dir, exist_ok=True)

    with open(f"{args.project_dir}logs/ddim/info.csv", 'r') as f:
        reader = list(csv.reader(f))

    if reader[0][0] == 'CIFAR10':
        with open("~/mia_diffusion/ddim/configs/cifar10.yml", "r") as f:
            config = yaml.safe_load(f)
        config = dict2namespace(config)
        train_loader, test_loader, info = mydataset.get_dataloader_cifar10(config.data.image_size, config.training.batch_size, int(reader[0][1]), int(reader[0][2]))
    elif reader[0][0] == 'CELEBA':
        with open("~/mia_diffusion/ddim/configs/celeba.yml", "r") as f:
            config = yaml.safe_load(f)
        config = dict2namespace(config)
        train_loader, test_loader, info = mydataset.get_dataloader_celeba(config.data.image_size, config.training.batch_size, int(reader[0][1]), int(reader[0][2]))

    if reader[0][3].strip() == 'cosine':
        config.diffusion.beta_schedule = 'cosine'
    elif reader[0][3].strip() == 'linear':
        config.diffusion.beta_schedule = 'linear'
    else:
        print("schedule error")
        return

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    config.device = device

    model = Model(config)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    states = torch.load(model_path)
    model.load_state_dict(states[0])

    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(model)
        ema_helper.load_state_dict(states[-1])
        ema_helper.ema(model)
    else:
        ema_helper = None
    
    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(device)
    model.eval()

    ### white-box
    rand = random.randint(0, data_skip-1)
    itr = -1
    for x, y in tqdm(train_loader):
        itr += 1
        if itr != 0 and itr % data_skip != rand:
            continue
        n = x.size(0)
        x = x.to(device)
        x = data_transform(config, x)
        b = betas

        loss_list = []
        for time in time_range:
            e = torch.randn_like(x)
            t = torch.full((n,), fill_value=time, dtype=torch.long).to(device)
            loss = loss_registry[config.model.type](model, x, t, e, b, keepdim=True).detach()
            loss_np = loss.to('cpu').detach().numpy().copy().flatten()
            loss_list.append(loss_np.tolist())
        if itr == 0:
            pos_loss = np.array(loss_list)
        else:
            pos_loss = np.concatenate([pos_loss, np.array(loss_list)], 1)

    itr = -1
    for x, y in tqdm(test_loader):
        itr += 1
        if itr != 0 and itr % data_skip != rand:
            continue
        n = x.size(0)
        x = x.to(device)
        x = data_transform(config, x)
        b = betas

        loss_list = []
        for time in time_range:
            e = torch.randn_like(x)
            t = torch.full((n,), fill_value=time, dtype=torch.long).to(device)
            loss = loss_registry[config.model.type](model, x, t, e, b, keepdim=True).detach()
            loss_np = loss.to('cpu').detach().numpy().copy().flatten()
            loss_list.append(loss_np.tolist())
        if itr == 0:
            neg_loss = np.array(loss_list)
        else:
            neg_loss = np.concatenate([neg_loss, np.array(loss_list)], 1)
    

    auc_list = []

    if os.path.exists(os.path.join(save_dir, 'hist_png')):
        shutil.rmtree(os.path.join(save_dir, 'hist_png'))
    os.makedirs(os.path.join(save_dir, 'hist_png'))
    max_auc = 0
    for i, time in enumerate(time_range):
        save_files(save_dir, ['pos_result'], [-pos_loss[i]])
        save_files(save_dir, ['neg_result'], [-neg_loss[i]])
        auc, accuracy, tpr = eval_roc.main('wb', save_dir, max_auc=max_auc)
        with open("~/mia_diffusion/result.txt", 'a') as f:
            print(f"Model: DDIM, Dataset: {reader[0][0]}, Data_num: {reader[0][1]}, Seed: {reader[0][2]}, Beta: {reader[0][3]}, Epoch: {ckpt}, Step: {time}, AUC: {auc:.4f}, Accuracy: {accuracy:.3f}, TPRat1%FPR: {tpr:.3f}", file=f)
        auc_list.append(auc)
        max_auc = max(max_auc, auc)
        max_loss = int(max(np.max(pos_loss[i]), np.max(neg_loss[i])))
        min_loss = int(min(np.min(pos_loss[i]), np.min(neg_loss[i])))
        bin = np.linspace(min_loss, max_loss, 50)
        plt.style.use('seaborn-whitegrid')
        plt.rcParams["font.size"] = 18
        kwargs = dict(bins=bin, alpha=0.5, histtype='stepfilled', edgecolor='k')
        plt.hist(pos_loss[i], label="member", **kwargs)
        plt.hist(neg_loss[i], label="non-member", **kwargs)
        #plt.title(f"step:{time}")
        plt.xlabel('value')
        plt.ylabel('num')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'hist_png', f'pos_neg_hist_step{time:04}.png'))
        plt.clf()
    
    files = sorted(glob.glob(os.path.join(save_dir, 'hist_png', '*.png')))
    images = list(map(lambda file : Image.open(file) , files))
    images[0].save(os.path.join(save_dir, 'pos_neg_hist.gif') , save_all = True , append_images = images[1:] , duration = 1000)

    plt.figure()
    plt.plot(time_range, auc_list, linestyle="solid", marker=".", label='auc')
    plt.title('AUC')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'auc.png'))
    plt.clf()

    with open(os.path.join(save_dir, 'auc.pkl'), 'wb') as f:
        pickle.dump(auc_list, f)


    for i, time in enumerate(time_range):
        mean = np.mean(pos_loss[i])
        if i == 0:
            pos_loss_mean = pos_loss[i] / mean
            neg_loss_mean = neg_loss[i] / mean
        else:
            pos_loss_mean += pos_loss[i] / mean
            neg_loss_mean += neg_loss[i] / mean

    save_files(save_dir, ['pos_result'], [-pos_loss_mean])
    save_files(save_dir, ['neg_result'], [-neg_loss_mean])
    auc, accuracy, tpr = eval_roc.main('wb', save_dir, max_auc=1)
    with open("~/mia_diffusion/result.txt", 'a') as f:
        print(f"Model: DDIM, Dataset: {reader[0][0]}, Data_num: {reader[0][1]}, Seed: {reader[0][2]}, Beta: {reader[0][3]}, Epoch: {ckpt}, Step: all, AUC: {auc:.4f}, Accuracy: {accuracy:.3f}, , TPRat1%FPR: {tpr:.3f}", file=f)
    max_loss = int(max(np.max(pos_loss_mean), np.max(neg_loss_mean)))
    min_loss = int(min(np.min(pos_loss_mean), np.min(neg_loss_mean)))
    bin = np.linspace(min_loss, max_loss, 50)
    plt.style.use('seaborn-whitegrid')
    kwargs = dict(bins=bin, alpha=0.5, histtype='stepfilled', edgecolor='k')
    plt.hist(pos_loss_mean, label="pos", **kwargs)
    plt.hist(neg_loss_mean, label="neg", **kwargs)
    plt.title(f"mean_sum")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'pos_neg_hist_mean.png'))
    plt.clf()

    '''
    mean_list = []
    betas = betas.to('cpu').detach().numpy().copy()
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod()
    for i, time in enumerate(range(time_skip-1,num_timesteps,time_skip)):
        pos_loss_scale = pos_loss[i] / (alphas_cumprod[time].sqrt() * 1000)
        mean = np.mean(pos_loss_scale)
        mean_list.append(mean)
    plt.figure()
    plt.plot(time_range, mean_list, label='mean')
    plt.title('Loss_scale')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'scale.png'))
    plt.clf()
    '''

    plt.close()

    return auc


if __name__ == '__main__':
    ckpts = [1000]
    aucs = []
    for ckpt in ckpts:
        auc = main(ckpt)
        aucs.append(auc)
    
    print(f"aucs:{aucs}")
