import torch
import numpy as np
import torch.nn as nn
from torch.autograd import no_grad
from Metric import computeTopNAccuracy_avg_useful_user

import ipdb

def full_ranking_double_check(epoch, model, rec_tensor, user_item_inter, mask_items, is_training, step, topk, prefix, writer=None):
    model.eval()
    if mask_items is not None:
        mask_items = torch.tensor(list(mask_items))
    with no_grad():               
        user_tensor = model.result[:model.num_user]
        item_tensor = model.result[model.num_user:]
        start_index = 0
        end_index = model.num_user if step==None else step
        all_index_of_rank_list = torch.LongTensor([])
        all_top_score_matrix = torch.LongTensor([])
        while end_index <= model.num_user and start_index < end_index:
            temp_user_tensor = user_tensor[start_index:end_index]
            score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
            if is_training is False: # mask training interactions
                for row, col in user_item_inter.items():
                    if row >= start_index and row < end_index:
                        row -= start_index
                        col = torch.LongTensor(list(col))-model.num_user
                        score_matrix[row][col] = -1e8
                if mask_items is not None:
                    score_matrix[:, mask_items-model.num_user] = -1e8

            top_score_matrix, index_of_rank_list = torch.topk(score_matrix, topk)
            all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu()), dim=0)
            all_top_score_matrix = torch.cat((all_top_score_matrix, top_score_matrix.cpu()),dim=0)
            start_index = end_index
            if end_index+step < model.num_user:
                end_index += step
            else:
                end_index = model.num_user
        return all_top_score_matrix, all_index_of_rank_list

def full_ranking_dropout(epoch, model, non_topN_mask, user_item_inter, mask_items, is_training, step, topk, drop='feature', prefix=None, writer=None, return_all=False):

    if mask_items is not None:
        mask_items = torch.tensor(list(mask_items))
    if drop:
        feature = model.encoder(drop=drop)
        model.result[model.feat_id + model.num_user] = feature[model.feat_id].data
    with no_grad():               
        user_tensor = model.result[:model.num_user]
        item_tensor = model.result[model.num_user:]
        start_index = 0
        end_index = model.num_user if step==None else step
        all_index_of_rank_list = torch.LongTensor([])
        all_top_score_matrix = torch.LongTensor([])
        all_score_matrix = torch.LongTensor([])
        while end_index <= model.num_user and start_index < end_index:
            temp_user_tensor = user_tensor[start_index:end_index]
            score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
            if is_training is False: # mask training interactions
                for row, col in user_item_inter.items():
                    if row >= start_index and row < end_index:
                        # mask all items behind topN for each user
                        if non_topN_mask is not None:
                            temp_col = non_topN_mask[row]
                            score_matrix[row-start_index][temp_col] = -1e8
                        row -= start_index
                        col = torch.LongTensor(list(col))-model.num_user
                        score_matrix[row][col] = -1e8

                if mask_items is not None:
                    score_matrix[:, mask_items-model.num_user] = -1e8
            
            top_score_matrix, index_of_rank_list = torch.topk(score_matrix, topk)
            all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu()), dim=0)
            all_top_score_matrix = torch.cat((all_top_score_matrix, top_score_matrix.cpu()),dim=0)
            all_score_matrix = torch.cat((all_score_matrix, score_matrix.cpu()), dim=0)
            start_index = end_index
            if end_index+step < model.num_user:
                end_index += step
            else:
                end_index = model.num_user
        if return_all:
            return all_score_matrix
        return all_top_score_matrix, all_index_of_rank_list

def rerank(data,scores,num_user,step,topk,mask_items):
    if mask_items is not None:
        mask_items = torch.tensor(list(mask_items))
    start_index = 0
    end_index = num_user if step==None else step
    all_index_of_rank_list = torch.LongTensor([])

    while end_index <= num_user and start_index < end_index:
        score_matrix = scores[start_index:end_index].clone()
        if mask_items is not None:
            score_matrix[:, mask_items-num_user] = -1e8
        _, index_of_rank_list = torch.topk(score_matrix, topk[-1])
        all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu()+num_user), dim=0)
        start_index = end_index
        if end_index+step < num_user:
            end_index += step
        else:
            end_index = num_user
    gt_list = [None for _ in range(num_user)]
    for u_id in data:
        gt_list[u_id] = data[u_id]
    results = computeTopNAccuracy_avg_useful_user(gt_list, all_index_of_rank_list, topk)
    return results

def compute_sim(anchor,all,emb):
    anchor_emb = emb[anchor]
    all_emb = emb[all]
    res = torch.mean(torch.mm(anchor_emb,all_emb.T),dim=1)
    return res

def rescore(DUIF_model, lc_graph, ui_graph, rec_graph):
    score_matrix = torch.zeros((DUIF_model.num_user, DUIF_model.num_item))
    item_emb = DUIF_model.result[DUIF_model.num_user:]
    for u_i in torch.nonzero(lc_graph):
        uidx, iidx = u_i.unsqueeze(dim=1)
        # similarity between cold item and history items
        his_iidx = torch.nonzero(ui_graph[u_i[0]]).squeeze(dim=1)
        score_matrix[uidx, iidx] += compute_sim(iidx.cuda(), his_iidx.cuda(), item_emb).cpu()
    return score_matrix