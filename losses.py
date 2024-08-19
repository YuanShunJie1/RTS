"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss
    
    



def custom_loss(logits, features, centroid1, centroid2):

    # 计算每个样本与centroid1的余弦相似性
    cos_sim1 = F.cosine_similarity(features, centroid1.unsqueeze(0), dim=1)
    
    # 计算每个样本与centroid2的余弦相似性
    cos_sim2 = F.cosine_similarity(features, centroid2.unsqueeze(0), dim=1)
    
    # 将样本划分到与其余弦相似性更高的簇心
    cluster_assignments = torch.where(cos_sim1 > cos_sim2, 0, 1)
    
    # 分别计算两个簇的分类损失
    loss_cluster1 = F.cross_entropy(logits[cluster_assignments == 0], cluster_assignments[cluster_assignments == 0])
    loss_cluster2 = F.cross_entropy(logits[cluster_assignments == 1], cluster_assignments[cluster_assignments == 1])
    
    # 合并损失
    total_loss = loss_cluster1 + loss_cluster2
    
    return total_loss



class ProtoLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ProtoLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, centroid1, centroid2, reduction='none'):
        
        # 计算每个样本与centroid1的余弦相似性
        cos_sim1 = F.cosine_similarity(features, centroid1.unsqueeze(0), dim=1)
        cos_sim2 = F.cosine_similarity(features, centroid2.unsqueeze(0), dim=1)
        
        mask1 = torch.where(cos_sim1 > cos_sim2, 1, 0)
        mask2 = torch.ones_like(mask1) - mask1


        value1 = torch.exp(torch.div(torch.matmul(features, centroid1.unsqueeze(1)), self.temperature)).flatten()
        value2 = torch.exp(torch.div(torch.matmul(features, centroid2.unsqueeze(1)), self.temperature)).flatten()
        
        if reduction == 'none':
            loss = -1 * torch.log(torch.div(value1*mask1 + value2*mask2, value1 + value2))
        elif reduction == 'mean':
            loss = torch.mean((-1 * torch.log(torch.div(value1*mask1 + value2*mask2, value1 + value2))))
        
        return loss



class NeighborConsistencyLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, eps=1e-8):
        super(NeighborConsistencyLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, features, logits, tao = -1):
        logits_softmax = F.softmax(logits, dim=1) + self.eps
        cos_sim = torch.matmul(features, features.T)

        # 寻找邻居
        cos_sim = cos_sim - torch.eye(cos_sim.size(0)).cuda()  # 排除自身
        neighbor_mask = cos_sim >= -1
        neighbor_mask = neighbor_mask.float()
        
        # 计算邻居的加权和
        neighbor_weights = cos_sim * neighbor_mask
        neighbor_weights = neighbor_weights / neighbor_weights.sum(dim=1, keepdim=True)
        weighted_sum_nb = torch.matmul(neighbor_weights, logits_softmax)

        # 计算KL散度
        kl_div_nb = F.kl_div(logits_softmax.log(), weighted_sum_nb, reduction='batchmean')

        return kl_div_nb


class CustomLoss(nn.Module):
    def __init__(self, tau=0.07):
        super(CustomLoss, self).__init__()
        self.tau = tau

    def forward(self, features, alpha):
        # features: [batch_size, dim]
        # alpha: similarity threshold
        features = F.normalize(features, p=2, dim=1)
        # 计算所有样本之间的相似性
        similarity_matrix = torch.matmul(features, features.t())  # [batch_size, batch_size]

        # 获取正样本掩码和负样本掩码
        positive_mask = similarity_matrix >= alpha
        negative_mask = similarity_matrix <  alpha

        # 用于避免对角线上的相似性（自相似）
        eye_mask = torch.eye(features.size(0), dtype=torch.bool, device=features.device)
        positive_mask = positive_mask & ~eye_mask
        negative_mask = negative_mask & ~eye_mask

        # 计算分子部分
        numerator = torch.sum(torch.exp(similarity_matrix / self.tau) * positive_mask, dim=1) + 1e-10

        # 计算分母部分
        denominator = numerator + torch.sum(torch.exp(similarity_matrix / self.tau) * negative_mask, dim=1) + 1e-10

        # 计算损失
        loss = -torch.log(numerator / denominator)

        return loss.mean()


# class NeighborConsistencyLoss(torch.nn.Module):
#     def __init__(self, temperature=0.07, eps=1e-8):
#         super(NeighborConsistencyLoss, self).__init__()
#         self.temperature = temperature
#         self.eps = eps

#     def forward(self, features, logits, tao):
#         logits_softmax = F.softmax(logits, dim=1) + self.eps
#         cos_sim = torch.matmul(features, features.T)

#         # 寻找邻居
#         # cos_sim = cos_sim - torch.eye(cos_sim.size(0)).cuda()  # 排除自身
#         neighbor_mask = cos_sim >= tao
#         neighbor_mask = neighbor_mask.float()
        
#         # 计算邻居的加权和
#         neighbor_weights = cos_sim * neighbor_mask
#         neighbor_weights = neighbor_weights / neighbor_weights.sum(dim=1, keepdim=True)
#         weighted_sum_nb = torch.matmul(neighbor_weights, logits_softmax)

#         # 计算KL散度
#         kl_div_nb = F.kl_div(logits_softmax.log(), weighted_sum_nb, reduction='batchmean')


#         # # 寻找邻居
#         # # cos_sim = cos_sim - torch.eye(cos_sim.size(0)).cuda()  # 排除自身
#         # non_neighbor_mask = cos_sim <= tao
#         # non_neighbor_mask = non_neighbor_mask.float()
        
#         # # 计算邻居的加权和
#         # non_neighbor_weights = cos_sim * non_neighbor_mask
#         # non_neighbor_weights = non_neighbor_weights / non_neighbor_weights.sum(dim=1, keepdim=True)
#         # non_weighted_sum_nb = torch.matmul(non_neighbor_weights, logits_softmax)

#         # # 计算KL散度
#         # non_kl_div_nb = F.kl_div(logits_softmax.log(), non_weighted_sum_nb, reduction='batchmean')


#         return kl_div_nb# - non_kl_div_nb

#         # kl_div_nb = F.kl_div((F.softmax(pred, dim=1)+eps).log(), F.softmax(targets, dim=1)+eps, reduction='none')


# class NeighborConsistecny(nn.Module):
#     def __init__(self, tao, temperature=0.07):
#         super(NeighborConsistecny, self).__init__()
#         self.temperature = temperature
#         self.tao = tao

#     def forward(self, features, logits):
#         logits_softmax = F.softmax(logits / self.temperature, dim=1)
#         cos_sim = torch.matmul(features, features.T, dim=1)

#         neighbors = []
#         for i in range(features.size(0)):
#             neighbor_indices = torch.where(cos_sim[i] >= self.tao)[0]
#             neighbor_indices = neighbor_indices[neighbor_indices != i]
#             neighbors.append(neighbor_indices.tolist())

#         weighted_sum_nb = torch.zeros_like(logits_softmax)
        
#         for i in range(features.size(0)):
#             _neighbors = neighbors[i]
#             _weights = cos_sim[i, _neighbors] / cos_sim[i, _neighbors].sum()
#             weighted_sum_nb[i] = torch.sum(_weights.unsqueeze(1) * logits_softmax[_neighbors], dim=0)
            
#         kl_div_nb = F.kl_div(logits_softmax.log(), weighted_sum_nb, reduction='batchmean')    

#         return kl_div_nb


        # non_neighbors = []
        # for i in range(cos_sim.size(0)):
        #     non_neighbor_indices = torch.where(cos_sim[i] < self.tao)[0]
        #     non_neighbor_indices = non_neighbor_indices[non_neighbor_indices != i]
        #     non_neighbors.append(non_neighbor_indices.tolist())
        

        # weighted_sum_non_nb = torch.zeros_like(logits_softmax)
        
        # for i in range(features.size(0)):
        #     _non_neighbors = non_neighbors[i]
        #     _weights = cos_sim[i, _non_neighbors] / cos_sim[i, _non_neighbors].sum()
        #     weighted_sum_non_nb[i] = torch.sum(_weights.unsqueeze(1) * logits_softmax[_non_neighbors], dim=0)
            
        # kl_div_non_nb = F.kl_div(logits_softmax.log(), weighted_sum_non_nb, reduction='batchmean')    
            
         

        # mask1 = torch.where(cos_sim1 > cos_sim2, 1, 0)
        # mask2 = torch.ones_like(mask1) - mask1


        # value1 = torch.exp(torch.div(torch.matmul(features, centroid1.unsqueeze(1)), self.temperature)).flatten()
        # value2 = torch.exp(torch.div(torch.matmul(features, centroid2.unsqueeze(1)), self.temperature)).flatten()
        
        # if reduction == 'none':
        #     loss = -1 * torch.log(torch.div(value1*mask1 + value2*mask2, value1 + value2))
        # elif reduction == 'mean':
        #     loss = torch.mean((-1 * torch.log(torch.div(value1*mask1 + value2*mask2, value1 + value2))))
        
        # return loss
