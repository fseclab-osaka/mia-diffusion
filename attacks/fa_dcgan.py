import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import argparse
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from tqdm.auto import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from tools.utils import *
import tools.eval_roc as eval_roc

sys.path.append('../')
import mydataset

sys.path.append('../dcgan/')
from dcgan import Discriminator

### Hyperparameters
BATCH_SIZE = 128


#############################################################################################################
# main
#############################################################################################################
def main(epoch):
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', default="~/mia_diffusion/dcgan/project/proj0/")
    args = parser.parse_args()

    model_path = f"{args.project_dir}model_epoch_{epoch}.pth"
    save_dir = f"{args.project_dir}samples_{epoch}/fa_result"

    os.makedirs(save_dir, exist_ok=True)

    ### load query images
    with open(f"{args.project_dir}info.csv", 'r') as f:
        reader = list(csv.reader(f))

    if reader[0][0] == 'cifar10':
        train_loader, test_loader, info = mydataset.get_dataloader_cifar10(32, BATCH_SIZE, int(reader[0][1]), int(reader[0][2]), normalize=True)
    elif reader[0][0] == 'celeba':
        train_loader, test_loader, info = mydataset.get_dataloader_celeba(64, BATCH_SIZE, int(reader[0][1]), int(reader[0][2]), normalize=True)

    ### white-box
    cuda = True
    device = torch.device("cuda:0" if cuda else "cpu")

    pos_results = []
    neg_results = []

    params = torch.load(model_path)['params']
    netD = Discriminator(params).to(device)
    netD.load_state_dict(torch.load(model_path)['discriminator'])
    netD.eval()

    # loop over training data
    for data, _ in tqdm(train_loader):
        real_cpu = data.to(device)
        output = netD(real_cpu).detach().cpu().numpy()
        pos_results.extend(output)


    # loop over test data
    for data, _ in tqdm(test_loader):
        real_cpu = data.to(device)
        output = netD(real_cpu).detach().cpu().numpy()
        neg_results.extend(output)

    print(f"num_pos_results:{len(pos_results)}")
    print(f"num_neg_results:{len(neg_results)}")

    plt.style.use('seaborn-whitegrid')
    plt.rcParams["font.size"] = 18
    bin = np.linspace(0, 1, 30)
    kwargs = dict(bins=bin, alpha=0.5, histtype='stepfilled', edgecolor='k')
    plt.hist(np.array(pos_results).flatten(), label="member", **kwargs)
    plt.hist(np.array(neg_results).flatten(), label="non-member", **kwargs)
    plt.xlabel('value')
    plt.ylabel('num')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pos_neg_hist.png'))
    plt.clf()

    save_files(save_dir, ['pos_result'], [pos_results])
    save_files(save_dir, ['neg_result'], [neg_results])

    auc, accuracy, tpr = eval_roc.main('wb', save_dir)

    with open("~/mia_diffusion/result.txt", 'a') as f:
            print(f"Model: DCGAN, Dataset: {reader[0][0]}, Data_num: {reader[0][1]}, Seed: {reader[0][2]}, Epoch: {epoch}, AUC: {auc:.4f}, Accuracy: {accuracy:.3f}, TPRat1%FPR: {tpr:.3f}", file=f)

    return auc


if __name__ == '__main__':
    epochs = [300]
    aucs = []
    for epoch in epochs:
        auc = main(epoch)
        aucs.append(auc)

    print(f"aucs:{aucs}")

