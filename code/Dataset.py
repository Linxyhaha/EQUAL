import time
import random
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import ipdb
import pdb
import itertools
import math

def data_load(dataset, has_v=True, has_a=True, has_t=True):

    dir_str = f'../data/{dataset}'

    # load interaction data
    train_data = np.load(dir_str+'/training_dict.npy', allow_pickle=True).item()
    val_data = np.load(dir_str+'/validation_dict.npy', allow_pickle=True).item()
    val_warm_data = np.load(dir_str+'/validation_warm_dict.npy', allow_pickle=True).item()
    val_cold_data = np.load(dir_str+'/validation_cold_dict.npy', allow_pickle=True).item()
    test_data = np.load(dir_str+'/testing_dict.npy', allow_pickle=True).item()
    test_warm_data = np.load(dir_str+'/testing_warm_dict.npy', allow_pickle=True).item()
    test_cold_data = np.load(dir_str+'/testing_cold_dict.npy', allow_pickle=True).item()

    # load the pretrained user/item embedding
    best_user_embedding = np.load(dir_str+'/user_embedding.npy', allow_pickle=True)
    best_item_embedding = np.load(dir_str+'/item_embedding.npy', allow_pickle=True)


    if dataset == 'micro-video':
        num_user = 21608 
        num_item = 64437
        num_warm_item = 56722
        pca_feat = np.load(dir_str + '/visual_feature_64.npy', allow_pickle=True)
        v_feat = np.zeros((num_item,pca_feat.shape[1])) # pca dim = 64
        for i_id in range(num_item):
            v_feat[i_id] = pca_feat[i_id]
            i_id += 1
        v_feat = torch.tensor(v_feat,dtype=torch.float).cuda()
        a_feat = None
        text_feat = np.load(dir_str + '/text_name_feature.npy', allow_pickle=True)
        t_feat = np.zeros((num_item,text_feat.shape[1]))
        for i_id in range(num_item):
            t_feat[i_id] = text_feat[i_id]
            i_id += 1
        t_feat = torch.tensor(t_feat,dtype=torch.float).cuda()

    elif dataset == 'amazon':
        num_user = 21607
        num_item = 93755
        num_warm_item = 75069
        pca_feat = np.load(dir_str + '/img_pca_map.npy', allow_pickle=True).item()
        v_feat = np.zeros((num_item,len(pca_feat[0]))) # pca dim = 64
        for i_id in pca_feat:
            v_feat[i_id] = np.array(pca_feat[i_id])
        v_feat = torch.tensor(v_feat,dtype=torch.float).cuda()
        a_feat = None
        t_feat = None
    
    elif dataset == 'kwai':
        num_user = 7010
        num_item = 86483
        num_warm_item = 74470
        pca_feat = np.load(dir_str + '/img_pca_map.npy', allow_pickle=True)
        v_feat = torch.tensor(pca_feat,dtype=torch.float).cuda()
        a_feat = None
        t_feat = None

    # item id <- item id + num_user
    for u_id in train_data:
        for i,i_id in enumerate(train_data[u_id]):
            train_data[u_id][i] = i_id + num_user
        for i,i_id in enumerate(val_data[u_id]):
            val_data[u_id][i] = i_id + num_user
        for i,i_id in enumerate(val_warm_data[u_id]):
            val_warm_data[u_id][i] = i_id + num_user
        for i,i_id in enumerate(val_cold_data[u_id]):
            val_cold_data[u_id][i] = i_id + num_user
        for i,i_id in enumerate(test_data[u_id]):
            test_data[u_id][i] = i_id + num_user
        for i,i_id in enumerate(test_warm_data[u_id]):
            test_warm_data[u_id][i] = i_id + num_user
        for i,i_id in enumerate(test_cold_data[u_id]):
            test_cold_data[u_id][i] = i_id + num_user

    return num_user, num_item, num_warm_item, best_user_embedding, best_item_embedding, train_data, val_data, val_warm_data, val_cold_data, test_data, test_warm_data, test_cold_data, v_feat, a_feat, t_feat


class TrainingDataset(Dataset):
    def __init__(self, num_user, num_item, user_item_dict, cold_set, train_data, num_neg, best_user_embedding, best_item_embedding, pos_ratio, neg_ratio):
        self.train_data = []
        self.aug_res_data = []
        for u_id in train_data:
            for i_id in train_data[u_id]:
                self.train_data.append([u_id,i_id])
                self.aug_res_data.append([[u_id,u_id],[i_id,i_id]])

        self.num_user = num_user
        self.num_item = num_item
        self.num_neg = num_neg
        self.user_item_dict = user_item_dict
        self.cold_set = cold_set
        self.all_set = set(range(num_user, num_user+num_item))-self.cold_set  # all warm item

        self.best_user_embedding = best_user_embedding
        self.best_item_embedding = best_item_embedding

        self.pos_ratio = pos_ratio
        self.neg_ratio = neg_ratio
        
        self.augmentation(train_data)

    def augmentation(self, train_data):
        # generate positive interpolated samples
        for u_id, i_ids in train_data.items():
            if len(i_ids) == 1: # mixup needs at least two interacted items
                continue
            user_embedding = torch.Tensor(self.best_user_embedding[u_id])
            new_ids = [i-self.num_user for i in i_ids]
            item_embedding = torch.Tensor(self.best_item_embedding[new_ids])
            scores = torch.matmul(user_embedding, item_embedding.t())
            _, index_min = torch.topk(scores, len(i_ids), largest=False)
            list1 = [i_ids[index_min[i]] for i in range(int(len(i_ids)/2))]
            item_combine = list(itertools.combinations(list1, 2))
            select_items = (random.sample(item_combine, min(len(item_combine), math.ceil(self.pos_ratio * len(i_ids)))))
            for sample in select_items:
                self.aug_res_data.append([[u_id,u_id], list(sample)])


    def __len__(self):
        return len(self.aug_res_data)

    def __getitem__(self, index):

        user, pos_item = self.aug_res_data[index]
        neg_item = random.sample(self.all_set-set(self.user_item_dict[user[0]]), self.num_neg)
        aug_negs = []

        # generate negative interpolated samples
        user_embedding = torch.Tensor(self.best_user_embedding[user[0]])
        new_ids = [i-self.num_user for i in neg_item]
        item_embedding = torch.Tensor(self.best_item_embedding[new_ids])
        scores = torch.matmul(user_embedding, item_embedding.t())
        _, index_min = torch.topk(scores, len(neg_item), largest=True)
        list1 = [neg_item[index_min[i]] for i in range(int(len(neg_item)/2))]
        item_combine = list(itertools.combinations(list1, 2))
        while len(aug_negs) < len(neg_item)*self.neg_ratio:
            combine_item = random.choice(item_combine)
            while combine_item in aug_negs:
                combine_item = random.choice(item_combine)
            aug_negs.append(combine_item)

        user_tensor = torch.LongTensor(user)
        item_tensor = torch.LongTensor(pos_item + neg_item)
        
        return user_tensor, item_tensor, np.array(aug_negs)
