import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import metrics
import pickle


############################################################################
# visualization functions
############################################################################
def plot_roc(pos_results, neg_results):
    labels = np.concatenate((np.zeros((len(neg_results),)), np.ones((len(pos_results),))))
    results = np.concatenate((neg_results, pos_results))
    fpr, tpr, threshold = metrics.roc_curve(labels, results, pos_label=1)
    auc = metrics.roc_auc_score(labels, results)
    ap = metrics.average_precision_score(labels, results)
    return fpr, tpr, threshold, auc, ap

'''
def plot_hist(pos_dist, neg_dist, save_file):
    plt.figure()
    plt.hist(pos_dist, bins=100, alpha=0.5, weights=np.zeros_like(pos_dist) + 1. / pos_dist.size, label='positive')
    plt.hist(neg_dist, bins=100, alpha=0.5, weights=np.zeros_like(neg_dist) + 1. / neg_dist.size, label='negative')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.xlabel('distance')
    plt.ylabel('normalized frequency')
    plt.savefig(save_file)
    plt.clf()
    plt.close()
'''

#############################################################################################################
# main
#############################################################################################################
def main(attack_type, result_load_dir, reference_load_dir=None, max_auc=0):
    pos_results = np.load(os.path.join(result_load_dir, 'pos_result.npy')).flatten()
    neg_results = np.load(os.path.join(result_load_dir, 'neg_result.npy')).flatten()

    ### plot roc curve
    if reference_load_dir is None:
        fpr, tpr, threshold, auc, ap = plot_roc(pos_results, neg_results)
        if auc > max_auc:
            plt.rcParams["font.size"] = 16
            #plt.plot(fpr, tpr, label='%s attack, auc=%.3f, ap=%.3f' % (attack_type, auc, ap))
            plt.plot(fpr, tpr, label='%s attack, auc=%.3f, ap=%.3f' % (attack_type, auc, ap))
            plt.plot(np.linspace(0,1,len(tpr)), np.linspace(0,1,len(fpr)), label='base')
            plt.loglog()
            with open("/home/t-matsumoto/temp/med6000/bb_ddim_cifar10_100.pkl", 'wb') as f:
                pickle.dump([fpr,tpr], f)
            plt.legend(loc='lower right')
            plt.xlabel('false positive')
            plt.ylabel('true positive')
            plt.title('ROC curve')
            plt.tight_layout()
            plt.savefig(os.path.join(result_load_dir, 'roc.png'))
            plt.clf()
            plt.close()
        print("The AUC ROC value of %s attack is: %.3f " % (attack_type, auc))
    
    '''
    else:
        pos_ref = np.load(os.path.join(reference_load_dir, 'pos_loss.npy'))
        neg_ref = np.load(os.path.join(reference_load_dir, 'neg_loss.npy'))

        num_pos_samples = np.minimum(len(pos_loss), len(pos_ref))
        num_neg_samples = np.minimum(len(neg_loss), len(neg_ref))

        try:
            pos_calibrate = pos_loss[:num_pos_samples] - pos_ref[:num_pos_samples]
            neg_calibrate = neg_loss[:num_neg_samples] - neg_ref[:num_neg_samples]

        except:
            pos_calibrate = pos_loss[:num_pos_samples] - pos_ref[:num_pos_samples, 0]
            neg_calibrate = neg_loss[:num_neg_samples] - neg_ref[:num_neg_samples, 0]

        fpr, tpr, threshold, auc, ap = plot_roc(pos_calibrate, neg_calibrate)
        plt.plot(fpr, tpr, label='calibrated %s attack, auc=%.3f, ap=%.3f' % (attack_type, auc, ap))
        plt.legend(loc='lower right')
        plt.xlabel('false positive')
        plt.ylabel('true positive')
        plt.title('ROC curve')
        plt.savefig(os.path.join(result_load_dir, 'roc.png'))
        plt.clf()
        plt.close()
        print("The AUC ROC value of calibrated %s attack is: %.3f " % (attack_type, auc))
    '''
    

    #np.set_printoptions(threshold=1000)
    threshold_med = np.mean(np.concatenate([pos_results, neg_results]))
    label = np.concatenate([np.ones(pos_results.shape[0], dtype="int8"), np.zeros(neg_results.shape[0], dtype="int8")])
    pred = np.concatenate([np.where(pos_results>threshold_med, 1, 0), np.where(neg_results>threshold_med, 1, 0)])
    accuracy = metrics.accuracy_score(label, pred)
    print(f"accuracy(%):{accuracy*100}")

    '''
    loss = np.concatenate([pos_results, neg_results])
    sort_idx = np.argsort(-loss)
    top_10_label = label[sort_idx[:10]]
    top_20_label = label[sort_idx[:20]]
    top_50_label = label[sort_idx[:50]]
    print(f"num_pos_in_top10:{np.count_nonzero(top_10_label == 1)}")
    print(f"num_pos_in_top20:{np.count_nonzero(top_20_label == 1)}")
    print(f"num_pos_in_top50:{np.count_nonzero(top_50_label == 1)}")
    '''


    '''
    label = np.concatenate([np.ones(pos_loss.shape[0], dtype="int8"), np.zeros(neg_loss.shape[0], dtype="int8")])
    loss = np.concatenate([-pos_loss, -neg_loss])
    results = np.stack([label, loss])
    accuracy = 1 - results[0][results[1].argsort()][-pos_loss.shape[0]:].mean()
    print(f"accuracy:{accuracy}")
    '''

    ### TPR at 1%FPR
    for i in range(len(fpr)):
        if fpr[i] > 0.01:
            tpr_at_fpr = tpr[i-1]*100
            print(f'TPR at 1% FPR: {tpr_at_fpr}')
            break

    return auc, accuracy, tpr_at_fpr


if __name__ == '__main__':
    main('fa',
    result_load_dir="/home/t-matsumoto/ddim/project3/samples_200000/fa_result",
    reference_load_dir="/home/t-matsumoto/ddim/project5/samples_200000/fa_result")
