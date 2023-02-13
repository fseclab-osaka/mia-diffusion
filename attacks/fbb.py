import numpy as np
import os
import sys
import argparse
import csv
from functools import partial
from tqdm.auto import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from tools.utils import *
import tools.eval_roc as eval_roc
from sklearn.neighbors import NearestNeighbors

import torch
import torchvision.utils as tvu

sys.path.append('../')
import mydataset

### Hyperparameters
K = 5
BATCH_SIZE = 100


#############################################################################################################
# main nearest neighbor search function
#############################################################################################################
def find_knn(nn_obj, query_imgs):
    '''
    :param nn_obj: Nearest Neighbor object
    :param query_imgs: query images
    :return:
        dist: distance between query samples to its KNNs among generated samples
        idx: index of the KNNs
    '''
    dist = []
    idx = []
    for x_batch, _ in tqdm(query_imgs):
        x_batch = np.reshape(x_batch, [BATCH_SIZE, -1])
        dist_batch, idx_batch = nn_obj.kneighbors(x_batch, K)
        dist.append(dist_batch)
        idx.append(idx_batch)
    
    print(len(dist))
    try:
        dist = np.concatenate(dist)
        idx = np.concatenate(idx)
    except:
        dist = np.array(dist)
        idx = np.array(idx)
    print(dist.shape)
    return dist, idx


def find_pred_z(gen_z, idx):
    '''
    :param gen_z: latent codes of the generated samples
    :param idx: index of the KNN
    :return:
        pred_z: predicted latent code
    '''
    pred_z = []
    for i in range(len(idx)):
        pred_z.append([gen_z[idx[i, nn]] for nn in range(K)])
    pred_z = np.array(pred_z)
    return pred_z


#############################################################################################################
# main
#############################################################################################################
def main(epoch):
    parser = argparse.ArgumentParser()
    model = "ddim"
    sampling_step = "20"
    parser.add_argument('--project_dir', default=f"~/mia_diffusion/{model}/project/proj0/")
    args = parser.parse_args()

    load_dir = os.path.join(args.project_dir, f"samples_{epoch}")
    save_dir = f"{load_dir}/fbb_result"
    os.makedirs(save_dir, exist_ok=True)

    ### load generated samples
    if model == "ddim":
        generate = np.load(os.path.join(load_dir, f'generated_{sampling_step}.npz'))
        print(f"generated_{sampling_step}.npz")
    else:
        generate = np.load(os.path.join(load_dir, 'generated.npz'))
    gen_imgs = generate['img_r01']
    #gen_z = generate['noise']
    print("gen_imgs" + str(gen_imgs.shape))

    '''
    plt.axis("off")
    plt.title("Generated Images")
    gen_imgs_tensor = torch.from_numpy(gen_imgs.astype(np.float32)).clone()
    result = np.transpose(tvu.make_grid(gen_imgs_tensor[:64], padding=2, normalize=False), (1,2,0))
    plt.imsave(os.path.join(save_dir, "sample.png"), result.to('cpu').detach().numpy().copy())
    plt.clf()
    '''

    gen_feature = np.reshape(gen_imgs, [len(gen_imgs), -1])
    gen_feature = 2. * gen_feature - 1.
    print("gen_feature" + str(gen_feature.shape))

    ### load query images
    if model == "ddim":
        args.project_dir = f"{args.project_dir}/logs/ddim/"
    with open(f"{args.project_dir}info.csv", 'r') as f:
        reader = list(csv.reader(f))

    if reader[0][0] == 'cifar10' or reader[0][0] == 'CIFAR10':
        train_loader, test_loader, info = mydataset.get_dataloader_cifar10(32, BATCH_SIZE, int(reader[0][1]), int(reader[0][2]), normalize=True)
    elif reader[0][0] == 'celeba' or reader[0][0] == 'CELEBA':
        train_loader, test_loader, info = mydataset.get_dataloader_celeba(64, BATCH_SIZE, int(reader[0][1]), int(reader[0][2]), normalize=True)
    
    '''
    tmp = train_loader.__iter__()
    img, _ = tmp.next()
    plt.axis("off")
    plt.title("Generated Images")
    print(f"img:{img.shape}")
    result = np.transpose(tvu.make_grid(img, padding=2, normalize=False), (1,2,0))
    plt.imsave(os.path.join(save_dir, "sample_train.png"), result.cpu().numpy())

    tmp = test_loader.__iter__()
    img, _ = tmp.next()
    plt.axis("off")
    plt.title("Generated Images")
    print(f"img:{img.shape}")
    result = np.transpose(tvu.make_grid(img, padding=2, normalize=False), (1,2,0))
    plt.imsave(os.path.join(save_dir, "sample_test.png"), result.cpu().numpy())
    '''

    ### nearest neighbor search
    nn_obj = NearestNeighbors(K, n_jobs=16)
    nn_obj.fit(gen_feature)

    ### positive query
    pos_loss, pos_idx = find_knn(nn_obj, train_loader)
    pos_loss = np.sum(pos_loss, axis=1)
    print(f"pos_loss:{pos_loss.shape}")
    '''
    set_pos = []
    for i in range(len(pos_loss)//100):
        set_pos.append(np.sum(pos_loss[i*100:(i+1)*100]))
    pos_loss = np.array(set_pos).flatten()
    print(f"pos_loss:{pos_loss.shape}")
    '''
    save_files(save_dir, ['pos_result'], [-pos_loss])
    #pos_z = find_pred_z(gen_z, pos_idx)
    #save_files(save_dir, ['pos_loss', 'pos_idx', 'pos_z'], [pos_loss, pos_idx, pos_z])

    ### negative query
    neg_loss, neg_idx = find_knn(nn_obj, test_loader)
    neg_loss = np.sum(neg_loss, axis=1)
    print(f"neg_loss:{neg_loss.shape}")
    '''
    set_neg = []
    for i in range(len(neg_loss)//100):
        set_neg.append(np.sum(neg_loss[i*100:(i+1)*100]))
    neg_loss = np.array(set_neg).flatten()
    print(f"neg_loss:{neg_loss.shape}")
    '''
    save_files(save_dir, ['neg_result'], [-neg_loss])
    #neg_z = find_pred_z(gen_z, neg_idx)
    #save_files(save_dir, ['neg_loss', 'neg_idx', 'neg_z'], [neg_loss, neg_idx, neg_z])

    max_loss = int(max(np.max(pos_loss), np.max(neg_loss)))
    min_loss = int(min(np.min(pos_loss), np.min(neg_loss)))
    bin = np.linspace(min_loss, max_loss, 50)
    plt.style.use('seaborn-whitegrid')
    kwargs = dict(bins=bin, alpha=0.5, histtype='stepfilled', edgecolor='k')
    plt.hist(pos_loss, label="pos", **kwargs)
    plt.hist(neg_loss, label="neg", **kwargs)
    plt.title(f"kneighbor distance")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'pos_neg_hist.png'))
    plt.clf()

    auc, accuracy, tpr = eval_roc.main('fbb', save_dir)

    with open("~/result.txt", 'a') as f:
        print(f"Model: {model}_fbb, Dataset: {reader[0][0]}, Data_num: {reader[0][1]}, Seed: {reader[0][2]}, Epoch: {epoch}, AUC: {auc:.4f}, Accuracy: {accuracy:.3f}, TPRat1%FPR: {tpr:.3f}", file=f)

    return auc


if __name__ == '__main__':
    epochs = [500]
    aucs = []
    for epoch in epochs:
        auc = main(epoch)
        aucs.append(auc)

    print(f"aucs:{aucs}")
