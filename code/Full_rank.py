from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import no_grad
import numpy as np
import ipdb
import pdb
from Metric import rank, full_accuracy, computeTopNAccuracy, computeTopNAccuracy_avg_useful_user

def full_ranking(epoch, model, data, user_item_inter, mask_items, is_training, step, topk , prefix, writer=None): 
    #print(prefix+' start...')
    model.eval()
    if mask_items is not None:
        mask_items = torch.tensor(list(mask_items))
    with no_grad():               
        all_index_of_rank_list = rank(model.num_user, user_item_inter, mask_items, model.result, is_training, step, topk[-1])
        #precision, recall, ndcg_score = full_accuracy(data, all_index_of_rank_list, user_item_inter, is_training, topk[-1])
        gt_list = [None for _ in range(model.num_user)]
        for u_id in data:
            gt_list[u_id] = data[u_id]
        #ipdb.set_trace()
        #results = computeTopNAccuracy_avg_useful_user(gt_list, all_index_of_rank_list, topk)
        results = computeTopNAccuracy_avg_useful_user(gt_list, all_index_of_rank_list, topk)
        return results

def sub_ranking(model, dataloader, gt_data, topk):
    model.eval()
    user_tensor = torch.LongTensor([])
    with no_grad():           
        all_index_of_rank_list = torch.zeros((model.num_user,topk[-1]),dtype=torch.int64)
        all_score = torch.zeros((model.num_user,1000)).cuda()
        true_idx_of_items = torch.zeros((model.num_user,1000),dtype=torch.int64)
        for user, items in dataloader:
            for i in range(len(user)):
                user_tensor = model.result[user[i]].view(1,-1) # (1,64)
                item_tensor = model.result[items[i]] # (1000,64)
                score = torch.matmul(user_tensor, item_tensor.t()).squeeze(0)
                all_score[user[i]] = score
                true_idx_of_items[user[i]] = items[i]
        _, fake_idx_of_rank_list = torch.topk(all_score, topk[-1])
        for i in range(len(all_index_of_rank_list)):
            all_index_of_rank_list[i] = true_idx_of_items[i][fake_idx_of_rank_list[i]]

        gt_list = [None for _ in range(model.num_user)]
        for u_id in gt_data:
            gt_list[u_id] = gt_data[u_id]
        results = computeTopNAccuracy_avg_useful_user(gt_list, all_index_of_rank_list, topk)
        return results


