from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import ipdb
import pdb
import copy
import itertools
import random
# from torch_geometric.utils import scatter_   # textual feature


##########################################################################

class CLCRec(torch.nn.Module):
    def __init__(self, warm_item, cold_item, num_user, num_item, num_warm_item, train_data, reg_weight, dim_E, v_feat, a_feat, t_feat, temp_value, num_neg, lr_lambda, is_word, alpha, mse_weight=0, num_sample=0.5, aug_mode='both'):
        super(CLCRec, self).__init__()
        self.num_user = num_user
        self.num_neg = num_neg
        self.train_data = train_data
        self.lr_lambda = lr_lambda
        self.reg_weight = reg_weight
        self.temp_value = temp_value
        self.is_word = is_word
        self.id_embedding = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_E))))
        self.dim_feat = 0
        self.num_sample = num_sample
        self.alpha = alpha
        self.mse_weight = mse_weight
        
        self.emb_id = list(range(num_user)) + list(warm_item)
        self.feat_id = torch.tensor([i_id-num_user for i_id in cold_item])

        self.result = torch.zeros((num_user+num_item,dim_E)).cuda()
        
        if v_feat is not None:
            self.v_feat = F.normalize(v_feat, dim=1)
            self.dim_feat += self.v_feat.size(1)
        else:
            self.v_feat = None
        
        if a_feat is not None:
            self.a_feat = F.normalize(a_feat, dim=1)
            self.dim_feat += self.a_feat.size(1)
        else:
            self.a_feat = None

        if t_feat is not None:
            if is_word:
                self.t_feat = nn.Parameter(nn.init.xavier_normal_(torch.rand((torch.max(t_feat[1]).item()+1, 128))))
                self.word_tensor = t_feat
            else:
                self.t_feat = F.normalize(t_feat, dim=1)
            self.dim_feat += self.t_feat.size(1)
        else:
            self.t_feat = None
        
        self.MLP = nn.Linear(dim_E, dim_E)

        self.encoder_layer1 = nn.Linear(self.dim_feat, 256)
        self.encoder_layer2 = nn.Linear(256, dim_E)
        
        self.bias = nn.Parameter(nn.init.kaiming_normal_(torch.rand((dim_E, 1))))
        self.att_sum_layer = nn.Linear(dim_E, dim_E)

        self.result = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_E))).cuda()

    def encoder(self, mask=None, drop=False):
        feature = torch.tensor([]).cuda()
        if self.v_feat is not None:
            feature = torch.cat((feature, self.v_feat), dim=1)
        if self.a_feat is not None:
            feature = torch.cat((feature, self.a_feat), dim=1)
        if self.t_feat is not None:
            if self.is_word:
                t_feat = F.normalize(torch.tensor(torch.scatter('mean', self.t_feat[self.word_tensor[1]], self.word_tensor[0]))).cuda()
                feature = torch.cat((feature, t_feat), dim=1)
            else:
                feature = torch.cat((feature, self.t_feat), dim=1)
        feature = F.leaky_relu(self.encoder_layer1(feature))
        feature = self.encoder_layer2(feature)
        
        # for inference
        if drop=='feature':
            feature = self.drop(feature)
        feature = F.leaky_relu(self.encoder_layer1(feature))
        if drop=='model':
            feature = self.drop(feature)

        return feature

    def mixup_raw(self, aug_sample1, aug_sample2, batch_size):
        feature = torch.tensor([]).cuda()
        if self.v_feat is not None:
            feature = torch.cat((feature, self.v_feat), dim=1)
        if self.t_feat is not None:
            if self.is_word:
                t_feat = F.normalize(torch.tensor(torch.scatter('mean', self.t_feat[self.word_tensor[1]], self.word_tensor[0]))).cuda()
                feature = torch.cat((feature, t_feat), dim=1)
            else:
                feature = torch.cat((feature, self.t_feat), dim=1)
        lam = np.random.beta(self.alpha, self.alpha)
        id_emb = lam * self.id_embedding[aug_sample1] + (1 - lam) * self.id_embedding[aug_sample2]
        feature = lam * feature[aug_sample1[batch_size:]-self.num_user] + (1-lam) * feature[aug_sample2[batch_size:]-self.num_user]
        feature = F.leaky_relu(self.encoder_layer1(feature))
        feature = self.encoder_layer2(feature)
        return id_emb[:batch_size,:], id_emb[batch_size:,:], feature

    def mixup_raw_ng(self, sample1, sample2):
        feature = torch.tensor([]).cuda()
        if self.v_feat is not None:
            feature = torch.cat((feature, self.v_feat), dim=1)
        if self.t_feat is not None:
            if self.is_word:
                t_feat = F.normalize(torch.tensor(torch.scatter('mean', self.t_feat[self.word_tensor[1]], self.word_tensor[0]))).cuda()
                feature = torch.cat((feature, t_feat), dim=1)
            else:
                feature = torch.cat((feature, self.t_feat), dim=1)
        lam = np.random.beta(self.alpha, self.alpha)
        id_emb = lam * self.id_embedding[sample1] + (1 - lam) * self.id_embedding[sample2]

        feature = lam * feature[sample1-self.num_user] + (1-lam) * feature[sample2-self.num_user]
        feature = F.leaky_relu(self.encoder_layer1(feature))
        feature = self.encoder_layer2(feature)
        return id_emb, feature

    def mixup(self, sample1, sample2, feature, batch_size):
        lam = np.random.beta(self.alpha, self.alpha)
        id_emb = lam * self.id_embedding[sample1] + (1 - lam) * self.id_embedding[sample2]
        feat_emb = lam * feature[sample1[batch_size:]-self.num_user] + (1-lam) * feature[sample2[batch_size:]-self.num_user]
        return id_emb[:batch_size,:], id_emb[batch_size:, :], feat_emb

    def mixup_ng(self, sample1, sample2, feature, batch_size):
        lam = np.random.beta(self.alpha, self.alpha)
        id_emb = lam * self.id_embedding[sample1] + (1 - lam) * self.id_embedding[sample2]
        feat_emb = lam * feature[sample1-self.num_user] + (1-lam) * feature[sample2-self.num_user]
        return id_emb, feat_emb

    def align_1(self, aug_sample1, aug_sample2, aug_negs, neg_item_tensor, neg_item_emb, feature, batch_size):
        '''
            description: alignment between feature representation and raw feature
            return: user, item embeddings, and mse loss 
            note: mse loss between raw feature and feature rep only for original samples
        '''
        aug_mask = aug_sample1[batch_size:]-aug_sample2[batch_size:]
        aug_mask[aug_mask!=0]=1
        aug_mask = aug_mask.unsqueeze(dim=1)
        # 1. obtain mixup of feature raw (see mixup_raw)
        feat_raw = torch.tensor([]).cuda()
        if self.v_feat is not None:
            feat_raw = torch.cat((feat_raw, self.v_feat), dim=1)
        if self.t_feat is not None:
            if self.is_word:
                t_feat = F.normalize(torch.tensor(torch.scatter('mean', self.t_feat[self.word_tensor[1]], self.word_tensor[0]))).cuda()
                feat_raw = torch.cat((feat_raw, t_feat), dim=1)
            else:
                feat_raw = torch.cat((feat_raw, self.t_feat), dim=1)
        lam = np.random.beta(self.alpha, self.alpha)
        mixup_raw = lam * feat_raw[aug_sample1[batch_size:]-self.num_user] + (1-lam) * feat_raw[aug_sample2[batch_size:]-self.num_user]
        # 2. obtain feature rep of 1
        mixup_raw_rep = F.leaky_relu(self.encoder_layer1(mixup_raw))
        mixup_raw_rep = self.encoder_layer2(mixup_raw_rep)
        # 3. obtain mixup of feature rep (self.encoder())
        mixup_rep = lam * feature[aug_sample1[batch_size:]-self.num_user] + (1-lam) * feature[aug_sample2[batch_size:]-self.num_user]
        mse_loss_1 = F.mse_loss(mixup_raw_rep*aug_mask, mixup_rep*aug_mask)

        # for negative samples
        sample1 = aug_negs[:,:,0] #(bs, self.neg_ratio*ng, 2)
        sample2 = aug_negs[:,:,1]
        mixup_raw = lam * feat_raw[sample1-self.num_user] + (1-lam) * feat_raw[sample2-self.num_user]
        mixup_raw_rep = F.leaky_relu(self.encoder_layer1(mixup_raw))
        mixup_raw_rep = self.encoder_layer2(mixup_raw_rep)
        mixup_rep = lam * feature[sample1-self.num_user] + (1-lam) * feature[sample2-self.num_user]
        mse_loss_2 = F.mse_loss(mixup_raw_rep.view(batch_size,-1), mixup_rep.view(batch_size,-1))

        user_emb = self.id_embedding[aug_sample1[:batch_size]]
        pos_item_emb = self.id_embedding[aug_sample1[batch_size:]]
        pos_item_feature = feature[aug_sample1[batch_size:]-self.num_user]
        neg_item_feature = feature[neg_item_tensor-self.num_user].view(batch_size,-1)

        return user_emb, pos_item_emb, pos_item_feature, neg_item_emb, neg_item_feature, mse_loss_1+mse_loss_2, aug_mask

    def align_2(self, aug_sample1, aug_sample2, aug_negs, neg_item_tensor, neg_item_emb, feature, batch_size):
        '''
            description: alignment between feature representation and CF representation
            return: mixup results for user embedding(id emb mix) and item embedding, and item feature representation embedding
        '''
        user_emb, pos_item_emb, pos_item_feature = self.mixup(aug_sample1, aug_sample2, feature, batch_size) #(bs, dim)
        neg_item_feature = feature[neg_item_tensor-self.num_user].view(batch_size, -1)

        # negative samples augmentation
        sample1 = aug_negs[:,:,0] 
        sample2 = aug_negs[:,:,1]
        emb, neg_feature_aug = self.mixup_ng(sample1, sample2, feature, batch_size)
        emb = emb.view(batch_size, -1)
        neg_feature_aug = neg_feature_aug.view(batch_size, -1)

        neg_item_emb[...,:emb.size(1)] = emb
        neg_item_feature[...,:neg_feature_aug.size(1)] = neg_feature_aug
        return user_emb, pos_item_emb, pos_item_feature, neg_item_emb, neg_item_feature

    def align_3(self, aug_sample1, aug_sample2, aug_negs, neg_item_tensor, neg_item_emb, feature, batch_size):
        '''
            description: alignment between raw feature and CF representation
            return: mixup results for user embedding(id emb mix) and item embedding, and item feature representation embedding
        '''
        user_emb, pos_item_emb, pos_item_feature = self.mixup_raw(aug_sample1, aug_sample2, batch_size)
        neg_item_feature = feature[neg_item_tensor-self.num_user].view(batch_size, -1)
        
        # negative samples augmentation
        sample1 = aug_negs[:,:,0] 
        sample2 = aug_negs[:,:,1]
        emb, neg_feature_aug = self.mixup_raw_ng(sample1, sample2) 
        emb = emb.view(batch_size, -1)
        neg_feature_aug = neg_feature_aug.view(batch_size, -1)

        neg_item_emb[...,:emb.size(1)] = emb
        neg_item_feature[...,:neg_feature_aug.size(1)] = neg_feature_aug
        return user_emb, pos_item_emb, pos_item_feature, neg_item_emb, neg_item_feature


    def compute_loss(self,user_emb, pos_item_emb, pos_item_feature, neg_item_emb, neg_item_feature, batch_size, mse_loss=0):
        user_embedding = user_emb.repeat(1, 1+self.num_neg).view(batch_size*(1+self.num_neg), -1)
        pos_item_embedding = pos_item_emb.repeat(1, 1+self.num_neg).view(batch_size*(1+self.num_neg), -1)

        all_item_embedding = torch.cat([pos_item_emb, neg_item_emb], dim=1).view(batch_size*(1+self.num_neg),-1)
        all_item_feat = torch.cat([pos_item_feature, neg_item_feature], dim=1).view(batch_size*(1+self.num_neg),-1)
        
        head_feat = F.normalize(all_item_feat, dim=1)
        head_embed = F.normalize(pos_item_embedding, dim=1)

        all_item_input = all_item_embedding.clone()
        rand_index = torch.randint(all_item_embedding.size(0), (int(all_item_embedding.size(0)*self.num_sample), )).cuda()
        all_item_input[rand_index] = all_item_feat[rand_index].clone()

        contrastive_loss_1 = self.loss_contrastive(head_embed, head_feat, self.temp_value)
        contrastive_loss_2 = self.loss_contrastive(user_embedding, all_item_input, self.temp_value)

        reg_loss = ((torch.sqrt((user_embedding**2).sum(1))).mean()+(torch.sqrt((all_item_embedding**2).sum(1))).mean())/2
        return contrastive_loss_1 * self.lr_lambda + (contrastive_loss_2) * (1-self.lr_lambda) + self.mse_weight * mse_loss, reg_loss

    def align_all(self,aug_sample1, aug_sample2, aug_negs, neg_item_tensor, neg_item_emb, feature, batch_size):

        self.aug_mask = torch.zeros((batch_size,1)).cuda()
        user_emb, pos_item_emb, pos_item_feature, neg_item_emb, neg_item_feature = self.align_3(aug_sample1, aug_sample2, aug_negs, neg_item_tensor, neg_item_emb, feature, batch_size)
        loss_3, reg_loss_3 = self.compute_loss(user_emb, pos_item_emb, pos_item_feature, neg_item_emb, neg_item_feature, batch_size)

        user_emb, pos_item_emb, pos_item_feature, neg_item_emb, neg_item_feature = self.align_2(aug_sample1, aug_sample2, aug_negs, neg_item_tensor, neg_item_emb, feature, batch_size)
        loss_2, reg_loss_2 = self.compute_loss(user_emb, pos_item_emb, pos_item_feature, neg_item_emb, neg_item_feature, batch_size)

        user_emb, pos_item_emb, pos_item_feature, neg_item_emb, neg_item_feature, mse_loss, self.aug_mask = self.align_1(aug_sample1, aug_sample2, aug_negs, neg_item_tensor, neg_item_emb, feature, batch_size)
        loss_1, reg_loss_1 = self.compute_loss(user_emb, pos_item_emb, pos_item_feature, neg_item_emb, neg_item_feature, batch_size, mse_loss)

        self.result = self.id_embedding.data
        self.result[self.feat_id + self.num_user] = feature[self.feat_id].data

        return (loss_1+loss_2+loss_3)/3, (reg_loss_1+reg_loss_2+reg_loss_3)/3

    def forward(self, user_tensor, item_tensor, aug_negs, align):
        self.align=align
        mse_loss = 0        
        batch_size = user_tensor.size(0)
        pos_aug_item1_tensor = item_tensor[:, 0] #(bs, )
        pos_aug_item2_tensor = item_tensor[:, 1] #(bs, )
        aug_sample1 = user_tensor[:,0] #(bs, )
        aug_sample2 = user_tensor[:,1] #(bs, )

        aug_sample1 = torch.cat([aug_sample1, pos_aug_item1_tensor]) #(2*bs, )
        aug_sample2 = torch.cat([aug_sample2, pos_aug_item2_tensor]) #(2*bs, )

        neg_item = item_tensor[:, 2:] #(bs, ng)
        neg_item_tensor = neg_item.reshape(-1, 1).squeeze(1)  #(bs*ng, )
        neg_item_emb = self.id_embedding[neg_item_tensor].view(batch_size, -1) #(bs, ng*dim)
        
        feature = self.encoder()
        if self.align == 1: # raw feature <-> feature rep
            user_emb, pos_item_emb, pos_item_feature, neg_item_emb, neg_item_feature, mse_loss, self.aug_mask = self.align_1(aug_sample1, aug_sample2, aug_negs, neg_item_tensor, neg_item_emb, feature, batch_size)
        elif self.align == 2: # feature rep <-> CF rep
            user_emb, pos_item_emb, pos_item_feature, neg_item_emb, neg_item_feature = self.align_2(aug_sample1, aug_sample2, aug_negs, neg_item_tensor, neg_item_emb, feature, batch_size)
        elif self.align == 3: # raw feature <-> CF rep
            user_emb, pos_item_emb, pos_item_feature, neg_item_emb, neg_item_feature = self.align_3(aug_sample1, aug_sample2, aug_negs, neg_item_tensor, neg_item_emb, feature, batch_size)
        elif self.align == 0:
            return self.align_all(aug_sample1, aug_sample2, aug_negs, neg_item_tensor, neg_item_emb, feature, batch_size)
        
        user_embedding = user_emb.repeat(1, 1+self.num_neg).view(batch_size*(1+self.num_neg), -1)
        pos_item_embedding = pos_item_emb.repeat(1, 1+self.num_neg).view(batch_size*(1+self.num_neg), -1)

        all_item_embedding = torch.cat([pos_item_emb, neg_item_emb], dim=1).view(batch_size*(1+self.num_neg),-1)
        all_item_feat = torch.cat([pos_item_feature, neg_item_feature], dim=1).view(batch_size*(1+self.num_neg),-1)
        
        head_feat = F.normalize(all_item_feat, dim=1)
        head_embed = F.normalize(pos_item_embedding, dim=1)

        all_item_input = all_item_embedding.clone()
        rand_index = torch.randint(all_item_embedding.size(0), (int(all_item_embedding.size(0)*self.num_sample), )).cuda()
        all_item_input[rand_index] = all_item_feat[rand_index].clone()

        contrastive_loss_1 = self.loss_contrastive(head_embed, head_feat, self.temp_value)
        contrastive_loss_2 = self.loss_contrastive(user_embedding, all_item_input, self.temp_value)

        reg_loss = ((torch.sqrt((user_embedding**2).sum(1))).mean()+(torch.sqrt((all_item_embedding**2).sum(1))).mean())/2

        self.result = self.id_embedding.data
        self.result[self.feat_id + self.num_user] = feature[self.feat_id].data

        return contrastive_loss_1 * self.lr_lambda + (contrastive_loss_2) * (1-self.lr_lambda) + self.mse_weight * mse_loss, reg_loss

    def loss_contrastive(self, tensor_anchor, tensor_all, temp_value):      
        all_score = torch.exp(torch.sum(tensor_anchor*tensor_all, dim=1)/temp_value).view(-1, 1+self.num_neg)
        all_score = all_score.view(-1, 1+self.num_neg)
        pos_score = all_score[:, 0]
        all_score = torch.sum(all_score, dim=1)
        self.mat = (1-pos_score/all_score).mean()
        contrastive_loss = -torch.log(pos_score / all_score)
        if self.align==1 or self.align==0:
            contrastive_loss = contrastive_loss * (1-self.aug_mask.squeeze(dim=1))
        return contrastive_loss.mean()

    def loss(self, user_tensor, item_tensor, aug_negs, align=2):
        contrastive_loss, reg_loss = self.forward(user_tensor, item_tensor, aug_negs, align)
        reg_loss = self.reg_weight * reg_loss
        return reg_loss+contrastive_loss, reg_loss
