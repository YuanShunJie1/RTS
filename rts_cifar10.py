from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from models.vgg_cifar import vgg16
from models.mobilenetv2 import MobileNetV2
# from models.resnet_cifar import SimpleCNN, CNN, MLPNet, MMLPNet

from sklearn.mixture import GaussianMixture
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

from poison_tool_cifar import get_backdoor_loader, get_test_loader
from dataloader_cifar import get_labeled_loader, get_unlabeled_loader

import ssl_tool_newest as stn
# from torchvision import transforms, datasets
# from torch.utils.data import random_split, DataLoader, Dataset

# from losses import SCELoss, ProtoLoss, NeighborConsistencyLoss, CustomLoss
# from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
import seaborn as sns

# from tools import TempDataset, tsne_visualize
from sklearn.metrics.pairwise import cosine_similarity
from losses import SCELoss

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--id', default=1, type=int)
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='/home/shunjie/codes/robust_training_against_backdoor/ours_new_box/rts/CIFAR10/cifar-10-batches-py', type=str, help='path to dataset')
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--num_workers', default=5, type=int)

# Trigger_type
# 'gridTrigger'          BadNets
# 'fourCornerTrigger'   
# 'trojanTrigger'        Trojaning attack on Neural Networks
# 'blendTrigger'         Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning
# 'signalTrigger'        A New Backdoor Attack in Cnns by Training Set Corruption without Label Poisoning
# 'CLTrigger'            Label-Consistent Backdoor Attacks
# 'smoothTrigger'        Rethinking the backdoor attacks’ triggers: A frequency perspective
# 'dynamicTrigger'       Input-aware dynamic backdoor attack
# 'nashvilleTrigger'     Spectral signatures in backdoor attacks
# 'onePixelTrigger'      WaNet-Imperceptible Warping-based Backdoor Attack

# backdoor attacks
parser.add_argument('--target_label', type=int, default=8, help='class of target label')
parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')
# SSL
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda_u', default=15, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema_decay', default=0.999, type=float)
parser.add_argument('--rampup_length', default=190, type=int)
parser.add_argument('--train_iteration', default=1024, type=int)

# 'squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger', 'signalTrigger', 'trojanTrigger', 'CLTrigger', 'dynamicTrigger', 'nashvilleTrigger', 'onePixelTrigger', 'wanetTrigger'

parser.add_argument('--trigger_type', type=str, default='gridTrigger', help='type of backdoor trigger')
parser.add_argument('--poison_rate', type=float, default=0.1, help='ratio of backdoor poisoned data')
parser.add_argument('--model_type', default='resnet18', type=str)
parser.add_argument('--resume_path', default='/home/shunjie/codes/robust_training_against_backdoor/ours_new_box/rts/results_images/', type=str, help='path to dataset')

parser.add_argument('--nn', default=20, type=int, help='number of neighbors')
parser.add_argument('--tcsp', default=0.1, type=float, help='target class selection proportion')
parser.add_argument('--nsp', default=0.25, type=float, help='neighbor selection proportion')
parser.add_argument('--q_prob', default=0.5, type=float, help='probability to add a mask')
parser.add_argument('--width', default=8, type=int, help='suqre size')
parser.add_argument('--csp', default=0.2, type=float, help='clean selection proportion')


args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

args.resume_path = args.resume_path + f'dataset={args.dataset}_net={args.model_type}_trigger_type={args.trigger_type}_poison_rate={args.poison_rate}_nn={args.nn}_tcsp={args.tcsp}_nsp={args.nsp}_csp={args.csp}_q_prob={args.q_prob}_width={args.width}'

train_data_bad, backdoor_data_loader = get_backdoor_loader(args)
clean_test_loader, bad_test_loader = get_test_loader(args)

dir_path = args.resume_path
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

stats1_log = open(os.path.join(dir_path,f'stats1.txt'),'w')
stats2_log = open(os.path.join(dir_path,f'stats2.txt'),'w')
stats3_log = open(os.path.join(dir_path,f'stats3.txt'),'w')
stats4_log = open(os.path.join(dir_path,f'stats4.txt'),'w')

benign_test_log = open(os.path.join(dir_path,f'benign_test.txt'),'w')
poison_test_log = open(os.path.join(dir_path,f'poison_test.txt'),'w')



def collect_loss(net, attack_target=args.target_label):
    net.eval()
    tcis, tcfs, tcls, tcbl = [], [], [], []
    all_features = []
    label2indices = {}
    with torch.no_grad():
        for batch_idx, (inputs, labels, index) in enumerate(backdoor_data_loader):      
            inputs, labels = inputs.cuda(), labels.cuda() 
            outputs, features, _ = net(inputs)   
            losses = F.cross_entropy(outputs, labels, reduction='none')
            for i in range(inputs.size(0)):
                if labels[i].item() == attack_target:
                    tcis.append(index[i].item())
                    tcfs.append(features[i].detach())
                    tcls.append(losses[i].detach().item())
                    if index[i].item() in train_data_bad.poison_indices:
                        tcbl.append(1)
                    else:
                        tcbl.append(0)
                else:
                    if labels[i].item() in label2indices:
                        label2indices[labels[i].item()].append(index[i].item())
                    else:
                        label2indices[labels[i].item()] = []
                        label2indices[labels[i].item()].append(index[i].item())
                    
                all_features.append(features[i].detach())
            
    box = torch.tensor(tcls)
    tcls = (box-box.min())/(box.max()-box.min())  
    tcfs = torch.stack(tcfs)
    all_features = torch.stack(all_features)
    
    return tcis, tcfs, tcls, tcbl, all_features, label2indices
    

def poison_train(net, num_epoch=10, wu_lr=0.002):
    print('Poison Net\n','lr',wu_lr)
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=wu_lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    
    for epoch in range(num_epoch):
        num_iter = (len(backdoor_data_loader.dataset)//(2*args.batch_size))+1
        for batch_idx, (inputs, labels, index) in enumerate(backdoor_data_loader):      
            inputs, labels = inputs.cuda(), labels.cuda() 
            optimizer.zero_grad()
            outputs, features, _ = net(inputs)   
            loss = F.cross_entropy(outputs, labels)
            loss.backward()  
            optimizer.step()
            
            sys.stdout.write('\r')
            sys.stdout.write('%s:%.4f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'%(args.dataset, args.poison_rate, args.trigger_type, epoch, num_epoch, batch_idx+1, num_iter, loss.item()))
            sys.stdout.flush()
        
        scheduler.step()
        
        evaluate(epoch,net,clean_test_loader,benign_test_log,clean=True)
        evaluate(epoch,net,bad_test_loader,poison_test_log,clean=False)  
        print('\n')


def evaluate(epoch,net,test_loader,test_log,clean=True,record=True):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _, _ = net(inputs)
            _, predicted = torch.max(outputs, 1)            
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    
    if clean:
        print("\n| Test Epoch #%d\t Clean Accuracy: %.2f%%" %(epoch,acc))
    else:
        print("\n| Test Epoch #%d\t Poison Accuracy: %.2f%%" %(epoch,acc))
    if record:
        test_log.write('Epoch:%d Accuracy:%.2f\n'%(epoch,acc))
        test_log.flush()  



def mixmatch(args, labeled_trainloader, unlabeled_trainloader, model, num_epochs, ssl_lr=0.02, forensics=False):
    train_criterion = stn.SemiLoss()
    optimizer = optim.SGD(model.parameters(), lr=ssl_lr, momentum=0.9, weight_decay=5e-4)
    start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        lr = ssl_lr
        if epoch > num_epochs / 2:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr  

        stn.ssl_train(args, labeled_trainloader, unlabeled_trainloader, model, optimizer, train_criterion, epoch, num_epochs, use_cuda=True, forensics=forensics)
        
        evaluate(epoch,net,clean_test_loader,benign_test_log,clean=True)
        evaluate(epoch,net,bad_test_loader,poison_test_log,clean=False)  
        

def _temp_acc_recall(prediction, stats_log):
    prediction = prediction.astype(int)
    ground_truth = np.zeros(len(train_data_bad))
    ground_truth[train_data_bad.poison_indices] = 1
    
    recall   = recall_score(ground_truth, prediction)
    accuracy = accuracy_score(ground_truth, prediction)
    
    # print('\nNumer of poison samples:%d'%(pred.sum()))
    print('Recall:%.3f'%(recall))
    print('Accuracy:%.3f\n'%(accuracy))
    
    stats_log.write('Recall:%.3f Accuracy:%.3f\n'%(recall, accuracy))
    stats_log.flush()
 

def acc_recall(pred, target_class_indices, stats_log):
    prediction   = np.zeros(len(train_data_bad))
    for i in range(len(target_class_indices)):
        prediction[target_class_indices[i]] = pred[i]

    prediction = prediction.astype(int)
    ground_truth = np.zeros(len(train_data_bad))
    ground_truth[train_data_bad.poison_indices] = 1
    
    recall   = recall_score(ground_truth, prediction)
    accuracy = accuracy_score(ground_truth, prediction)
    
    print('\nNumer of poison samples:%d'%(pred.sum()))
    print('Recall:%.3f'%(recall))
    print('Accuracy:%.3f\n'%(accuracy))
    
    stats_log.write('Numer of poison samples:%d Recall:%.3f Accuracy:%.3f\n'%(pred.sum(), recall, accuracy))
    stats_log.flush()

    return prediction

# --model_type
def create_model(num_classes=args.num_class):
    if args.model_type == 'resnet18':
        model = ResNet18(num_classes=num_classes)
    if args.model_type == 'vgg16':
        model = vgg16(num_classes=num_classes)   
    if args.model_type == 'MobileNetV2':
        model = MobileNetV2(num_classes=num_classes)   
    model = model.cuda()
    return model


def backdoor_class_detection():
    print('Backdoor Class Detection\n')
    net = create_model(num_classes=args.num_class)
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9, weight_decay=2)
    
    for epoch in range(10):
        num_iter = (len(backdoor_data_loader.dataset)//(2*args.batch_size))+1
        for batch_idx, (inputs, labels, index) in enumerate(backdoor_data_loader):      
            inputs, labels = inputs.cuda(), labels.cuda() 
            optimizer.zero_grad()
            outputs, features, _ = net(inputs)   
            
            loss = F.cross_entropy(outputs, labels)
            
            loss.backward()  
            optimizer.step()
            
            sys.stdout.write('\r')
            sys.stdout.write('%s:%.4f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'%(args.dataset, args.poison_rate, args.trigger_type, epoch, 10, batch_idx+1, num_iter, loss.item()))
            sys.stdout.flush()
            
        evaluate(epoch,net,clean_test_loader,benign_test_log,clean=True)
        evaluate(epoch,net,bad_test_loader,poison_test_log,clean=False)  
    
    target = torch.argmax(outputs[0]).item()
    return target


def tsne_visualize(args, data, epoch, path='tsne_kmeans'):
    emb_all = data['emb_all']
    emb_benign_first= data['emb_benign_first']
    emb_poison_pred = data['emb_poison_second']
    emb_benign_pred = data['emb_benign_second']
    emb_poison_true = data['emb_poison_true']
    emb_benign_true = data['emb_benign_true']
    emb_benign_center = np.expand_dims(data['emb_benign_center'], axis=0)
    
    emb_benign_third = data['emb_benign_third']
    emb_poison_third = data['emb_poison_third']
    
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    sns.scatterplot(x=emb_all[:, 0], y=emb_all[:, 1], color='grey', label='Points', s=10, ax=axes[0])
    sns.scatterplot(x=emb_benign_first[:, 0], y=emb_benign_first[:, 1], color='#e1812c', label='Benign', s=10, ax=axes[0])
    sns.scatterplot(x=emb_benign_center[:, 0], y=emb_benign_center[:, 1], color='red', label='Center', s=10, ax=axes[0])
    axes[0].set_title('Confident Sample Selection')
    axes[0].legend(loc='lower right')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')
    axes[0].set_aspect('equal')  

    
    sns.scatterplot(x=emb_poison_pred[:, 0], y=emb_poison_pred[:, 1], color='#3274a1', label='Poison', s=10, ax=axes[1])
    sns.scatterplot(x=emb_benign_pred[:, 0], y=emb_benign_pred[:, 1], color='#e1812c', label='Benign', s=10, ax=axes[1])
    axes[1].set_title('Clean Sample Selection')
    axes[1].legend(loc='lower right')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    axes[1].set_aspect('equal')  


    sns.scatterplot(x=emb_poison_third[:, 0], y=emb_poison_third[:, 1], color='#3274a1', label='Poison', s=10, ax=axes[2])
    sns.scatterplot(x=emb_benign_third[:, 0], y=emb_benign_third[:, 1], color='#e1812c', label='Benign', s=10, ax=axes[2])
    axes[2].set_title('Neighborhood Backdoor Filtration')
    axes[2].legend(loc='lower right')
    axes[2].set_xlabel('')
    axes[2].set_ylabel('')
    axes[2].set_aspect('equal')  


    sns.scatterplot(x=emb_poison_true[:, 0], y=emb_poison_true[:, 1], color='#3274a1', label='Poison', s=10, ax=axes[3])
    sns.scatterplot(x=emb_benign_true[:, 0], y=emb_benign_true[:, 1], color='#e1812c', label='Benign', s=10, ax=axes[3])
    axes[3].set_title('Ground Truth')
    axes[3].legend(loc='lower right')
    axes[3].set_xlabel('')
    axes[3].set_ylabel('')
    axes[3].set_aspect('equal')  


    file_path_svg = os.path.join(args.resume_path, f'dataset={args.dataset}_at={args.trigger_type}_ar={args.poison_rate}_epoch={epoch}_clustering.svg')

    plt.savefig(file_path_svg, bbox_inches='tight')
    drawing = svg2rlg(file_path_svg)
    file_path_pdf = os.path.join(dir_path, f'dataset={args.dataset}_at={args.trigger_type}_ar={args.poison_rate}_epoch={epoch}_clustering.pdf')
    renderPDF.drawToFile(drawing, file_path_pdf)

    plt.close()
    os.remove(file_path_svg)
    


def count_clean_sample_selection_accuracy(selected_clean_indices):
    correct = 0
    
    for ind in selected_clean_indices:
        if ind not in train_data_bad.poison_indices:
            correct = correct + 1
    
    stats1_log.write('Numer of clean samples:%d Accuracy:%.3f\n'%(len(selected_clean_indices), correct/len(selected_clean_indices)))
    stats1_log.flush()


def neighbor_filter_iterative(neighbors, prediction, nn, nsp):
    stack = [i for i in range(len(prediction)) if not prediction[i]]

    while stack:
        i = stack.pop()
        if prediction[i]:
            continue
        
        similar_indices = neighbors[i]
        if sum(prediction[similar_indices]) >= int(nn * nsp):
            prediction[i] = True
            stack.extend(similar_indices)

def neighbor_filter(i, neighbors, prediction, nn, nsp):
    if prediction[i] == True:
        return   
    if prediction[i] == False:
        similar_indices = neighbors[i]
        if sum(prediction[similar_indices]) < int(nn*nsp):
            return 
    
        prediction[i] = True
        for nb in similar_indices:
            neighbor_filter(nb, neighbors, prediction, nn, nsp)



def sample_selection(num_epoch=10,nn=50,tcsp=0.05,nsp=0.5):
    net = create_model(num_classes=args.num_class)
    poison_train(net, num_epoch=num_epoch, wu_lr=0.002)
    
    # 收集损失
    target_class_indices, target_class_features, target_class_losses, target_class_backdoor_labels, all_features, label2indices = collect_loss(net, attack_target=args.target_label)
    all_features = all_features.cuda()
    
    # 选取部分大损失样本作为干净样本并计算簇心
    num_clean = int(len(target_class_indices) * tcsp)
    _, selected_confident_indices = torch.topk(target_class_losses, k=num_clean)
    selected_confident_indices = selected_confident_indices.numpy()
    
    cluster_benign = target_class_features[selected_confident_indices]
    center_benign  = cluster_benign.mean(dim=0)
    
    prediction_first = np.ones(len(target_class_indices))
    prediction_first[selected_confident_indices] = 0
    
    count_clean_sample_selection_accuracy(selected_confident_indices)
    
    similarities = F.cosine_similarity(target_class_features, center_benign.unsqueeze(0), dim=1)
    similarities = similarities.cpu()
    num_clean = int(len(target_class_indices) * args.csp)
    _, selected_clean_indices = torch.topk(similarities, k=num_clean)
    prediction_second = np.ones(len(target_class_indices))
    prediction_second[selected_clean_indices] = 0
    prediction_second[selected_confident_indices] = 0
    prediction_third  = prediction_second.copy()

    #根据邻域相似性质进行过滤
    cos_sim_mat = cosine_similarity(target_class_features.cpu().numpy())
    neighbors_dict = {}
    
    for i in range(cos_sim_mat.shape[0]):
        if prediction_third[i] == False:
            similar_indices = np.argsort(cos_sim_mat[i])[-nn:-1]
            neighbors_dict[i] = similar_indices
            
            if sum(prediction_third[similar_indices]) > int(nn*0.5):
                prediction_third[i] = True
    
    _temp_count = len(prediction_third) - np.sum(prediction_third)
    _temp_count = int(_temp_count)
    _box = []
    
    for label in label2indices:
        indices = np.array(label2indices[label],dtype=np.int64)
        _temp_features = all_features[indices]
        _temp_center   = _temp_features.mean(dim=0)
        _temp_similarities = F.cosine_similarity(_temp_features, _temp_center.unsqueeze(0), dim=1)
        _temp_similarities= _temp_similarities.cpu()
        _, selected_clean_indices = torch.topk(_temp_similarities, k=_temp_count)
        
        _box = _box + selected_clean_indices.tolist()
    
    # prediction_final = prediction_third.copy()
    # for i in range()
    prediction_final = np.ones(len(train_data_bad),dtype=np.int64)
    prediction_final[_box] = 0
    for i in range(len(target_class_indices)):
        prediction_final[target_class_indices[i]] = prediction_third[i]
    
    # prediction_final[_box] = 0
    
    acc_recall(prediction_second, target_class_indices, stats2_log)
    prediction_third_all = acc_recall(prediction_third, target_class_indices, stats3_log)
    _temp_acc_recall(prediction_final, stats4_log)
    
    return prediction_third_all, prediction_final


# net, clean_trainloader, num_epoch=50,lr=0.02
def train_with_clean_data(net, dataloader, num_epoch=50, lr=0.02,use_sce=True):
    
    alpha = 1.0
    beta  = 1.0
    criterion = SCELoss(alpha=alpha, beta=beta, num_classes=(args.num_class))
    
    print('Train Net\n')
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)

    for epoch in range(num_epoch):
        num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
        for batch_idx, (inputs, _, _, _, labels, out_index) in enumerate(dataloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs, _, _ = net(inputs)    
            
            if use_sce:
                loss = criterion(outputs, labels)
            else:
                loss = F.cross_entropy(outputs, labels)

            loss.backward()
            optimizer.step() 
      
            sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'%(args.dataset, args.poison_rate, args.trigger_type, epoch, num_epoch, batch_idx+1, num_iter, loss.item()))
            sys.stdout.flush()
            
        evaluate(epoch,net,clean_test_loader,benign_test_log,clean=True)
        evaluate(epoch,net,bad_test_loader,poison_test_log,clean=False)   
        scheduler.step()



# train_with_clean_data(net, clean_trainloader, num_epoch=30, lr=0.002)

if __name__ == '__main__':
    cudnn.benchmark = True
    target = backdoor_class_detection()
    
    net = create_model(num_classes=args.num_class)
    prediction, prediction_final = sample_selection(num_epoch=20,nn=args.nn,tcsp=args.tcsp, nsp=args.nsp)

    net = create_model(num_classes=args.num_class)
    # 重构均衡半监督学习训练集
    _, clean_trainloader  = get_labeled_loader(args, train_data_bad, pred=prediction_final, use_ssl=True)
    _, poison_trainloader = get_unlabeled_loader(args, train_data_bad, pred=prediction_final)
    mixmatch(args, clean_trainloader, poison_trainloader, net, 100, ssl_lr=0.002)
    
    # 
    _, clean_trainloader  = get_labeled_loader(args, train_data_bad, pred=prediction, use_ssl=True)
    _, poison_trainloader = get_unlabeled_loader(args, train_data_bad, pred=prediction)
    mixmatch(args, clean_trainloader, poison_trainloader, net, 100, ssl_lr=0.002)



# 1、后门类别检测
# 2、基于损失采集小部分干净样本，并获取干净样本中心 C_c
# 3、计算所有样本到C_c的余弦相似性，并将其进行GMM聚类得到相似性较高的类别，作为干净样本
# 4、后门压制的半监督学习


    