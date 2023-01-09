import argparse
import os
import time
from xml.etree.ElementInclude import default_loader
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import random
from Dataset import TrainingDataset, data_load, SubDataset
from model_CLCRec import CLCRec
from model_CBPR import CBPR_net
from model_GAR import GAR
from torch.utils.data import DataLoader
from Train import train
from Full_rank import full_ranking
from Metric import print_results
from IAM_util import full_ranking_double_check, full_ranking_dropout, rerank, rescore

###############################248###########################################

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Seed init.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--data_path', default='huawei', help='Dataset path')
    parser.add_argument('--save_file', default='', help='Filename')

    parser.add_argument('--PATH_weight_load', default=None, help='Loading weight filename.')
    parser.add_argument('--PATH_weight_save', default=None, help='Writing weight filename.')
    parser.add_argument('--prefix', default='', help='Prefix of save_file.')

    parser.add_argument('--l_r', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--lr_lambda', type=float, default=1, help='Weight loss one.')
    parser.add_argument('--reg_weight', type=float, default=1e-1, help='Weight decay.')
    parser.add_argument('--temp_value', type=float, default=1, help='Contrastive temp_value.')
    parser.add_argument('--model_name', default='SSL', help='Model Name.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--num_neg', type=int, default=512, help='Negative size.')
    parser.add_argument('--num_epoch', type=int, default=200, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=1, help='Workers number.')
    parser.add_argument('--num_sample', type=float, default=0.5, help='Workers number.')

    parser.add_argument('--dim_E', type=int, default=64, help='Embedding dimension.')
    parser.add_argument("--topK", default='[10, 20, 50, 100]', help="the recommended item num")
    parser.add_argument('--step', type=int, default=2000, help='Workers number.')
    
    parser.add_argument('--has_v', default='False', help='Has Visual Features.')
    parser.add_argument('--has_a', default='False', help='Has Acoustic Features.')
    parser.add_argument('--has_t', default='False', help='Has Textual Features.')

    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--save_path', default='./models/', help='model save path')
    parser.add_argument('--inference',action='store_true', help='only inference stage')
    parser.add_argument('--ckpt', type=str, help='pretrained model path')

    # mix up
    parser.add_argument('--pos_ratio', type=float, default=0.2, help='ratio for positive items')
    parser.add_argument('--neg_ratio', type=float, default=0.2, help='ratio for negative items')

    # rerank
    parser.add_argument('--drop_obj', default='feature', help='feature-based drop or feature encoder drop')
    parser.add_argument('--dropout', default='[0,0.05,0.1,0.15,0.2]', help='dropout ratio of uncertainty testing')
    parser.add_argument('--var_ratio', type=float, default=0.05, help='threshold of variance value')

    parser.add_argument('--rerank_model', type=str, default='DUIF', help='which model to use for rescoring')
    parser.add_argument('--DUIF_ckpt', type=str, help='pretrained DUIF model path')
    parser.add_argument('--theta', type=float, default=0.5, help='weight of original model in score fusion')
    parser.add_argument('--topN', type=int, default=20, help='topN cold items for re-scoring')

    parser.add_argument('--save_var', action='store_true')
    parser.add_argument('--log_name', default='log', help='log name')
    parser.add_argument('--backmodel', default='CLCRec', help='pre-trained back model name, e.g., CLCRec, GAR')
    args = parser.parse_args()
    return args

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

if __name__ == '__main__':
    args = init()
    print(args)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    ##########################################################################################################################################
    data_path = args.data_path
    save_file_name = args.save_file

    learning_rate = args.l_r
    lr_lambda = args.lr_lambda
    reg_weight = args.reg_weight
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epoch = args.num_epoch
    num_neg = args.num_neg
    num_sample = args.num_sample
    topK = eval(args.topK)
    prefix = args.prefix
    model_name = args.model_name
    temp_value = args.temp_value
    step = args.step
    topN = args.topN
    args.dropout = eval(args.dropout)

    has_v = True if args.has_v == 'True' else False
    has_a = True if args.has_a == 'True' else False
    has_t = True if args.has_t == 'True' else False

    dim_E = args.dim_E
    is_word = True if data_path == 'tiktok' else False
    writer = None

    ##########################################################################################################################################
    print('Data loading ...')
    num_user, num_item, num_warm_item, best_user_embedding, best_item_feature, best_item_embedding, train_data, val_data, val_warm_data, val_cold_data, test_data, test_warm_data, test_cold_data, v_feat, a_feat, t_feat = data_load(data_path)
    dir_str = f'../../../data/{data_path}'
        
    user_item_all_dict = {}
    train_dict = {}
    tv_dict = {}
    for u_id in train_data:
        user_item_all_dict[u_id] = train_data[u_id] + val_data[u_id] + test_data[u_id]
        train_dict[u_id] = train_data[u_id]
        tv_dict[u_id] = train_data[u_id] + val_data[u_id]
    warm_item = torch.tensor(list(np.load(dir_str + '/warm_item.npy',allow_pickle=True).item()))
    cold_item = torch.tensor(list(np.load(dir_str + '/cold_item.npy',allow_pickle=True).item()))
    warm_item = set([i_id + num_user for i_id in warm_item])    # item id = item_id_org + user num
    cold_item = set([i_id + num_user for i_id in cold_item])

    train_dataset = TrainingDataset(num_user, num_item, user_item_all_dict, cold_item, train_data, num_neg, best_user_embedding, best_item_feature, best_item_embedding, beta, augment, aug_mode)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)

    test_cold = np.load(dir_str+'/testing_cold_dict.npy', allow_pickle=True).item()
    print('Data has been loaded.')

    # infer on validation
    if args.log_name=='valid':
        test_data = val_data
        test_warm_data = val_warm_data
        test_cold_data = val_cold_data
        tv_dict = train_dict
    #=======================================================================================================#
    #                                               Inference                                               #
    #=======================================================================================================#
    # inference stage
    model = torch.load(args.ckpt)
    model.num_item = num_item
    model.eval()
    print('Model has been loaded.')

    if args.inference:
        with torch.no_grad():
            test_result = full_ranking(0, model, test_data, tv_dict, None, False, step, topK, 'Test/', writer)
            test_result_warm = full_ranking(0, model, test_warm_data, tv_dict, cold_item, False, step, topK, 'Test/warm_', writer)
            test_result_cold = full_ranking(0, model, test_cold_data, tv_dict, warm_item, False, step, topK, 'Test/cold_', writer)
            print('==='*18)
            print('All')
            print_results(None,None,test_result)
            print('Warm')
            print_results(None,None,test_result_warm)
            print('Cold')
            print_results(None,None,test_result_cold)
            print('==='*18)

    #=======================================================================================================#
    #                                      Find Low Confidence Samples                                      #
    #=======================================================================================================#
    with torch.no_grad():
        topN_scores_org, topN_rank = full_ranking_double_check(0, model, test_cold_data, tv_dict, warm_item, False, step, topN, 'Test/cold_', writer)
        all_scores_org = full_ranking_dropout(0, model, None, tv_dict, None, False, step, topN, False, 'Test/cold_', writer, True)

    # create non-topN item mask for each user
    all_item_list = list(range(model.num_item))
    mask_nontopN = {}
    for u_id in range(model.num_user):
        mask_nontopN[u_id] = torch.LongTensor(list(set(all_item_list)-set(topN_rank[u_id].tolist())))

    score_dropouts = torch.Tensor([])
    for drop_ratio in args.dropout:
        model.drop = nn.Dropout(drop_ratio)
        model.apply(apply_dropout)
        with torch.no_grad():
            topN_scores, topN_rank = full_ranking_dropout(0, model, mask_nontopN, tv_dict, warm_item, False, step, topN, args.drop_obj, 'Test/cold_', writer)
            #print(topN_scores[0])
            score_dropouts = torch.cat([score_dropouts, topN_scores.unsqueeze(0)], dim=0)
    variance = torch.var(score_dropouts,dim=0)
    if args.save_var:
        np.save(f'uncertain_variance_{args.drop_obj}_{args.dropout}.npy', variance)

    score_list = []
    for u_id in test_cold:
        if len(test_cold[u_id]):
            score_list.extend(variance[u_id])
    score_list = sorted(score_list,reverse=True)
    th = score_list[min(len(score_list)-1,int(len(score_list) * args.var_ratio))]

    lowconf_item = {}
    for u_id in test_cold:
        if len(test_cold[u_id]):
            i_indices = torch.nonzero(variance[u_id]>th)
            i_indices = i_indices.squeeze() if len(i_indices.shape)>1 else i_indices.unsqueeze(dim=0)
            i_indices = topN_rank[u_id][i_indices]
            i_indices = i_indices if len(i_indices.shape) else i_indices.unsqueeze(dim=0)
            if len(i_indices):
                lowconf_item[u_id] = i_indices 
    #=======================================================================================================#
    #                                                Re-rank                                                #
    #=======================================================================================================#
    # 1. construct a mask for DUIF, keeping the scores of rescore_item for each user
    # 2. do inference of DUIF, get the score_matrix
    # 3. fusion scores
    # matrix_org * (1-mask) + mask * (matrix_org + DUIF_matrix) / 2 

    # 1. construct mask
    lowconf_mask = torch.zeros([model.num_user, model.num_item])
    for u_id in lowconf_item:
        lowconf_mask[u_id][lowconf_item[u_id]] = 1

    # 2. do inference of DUIF, get the score_matrix
    if args.rerank_model=='DUIF':
        DUIF_model = torch.load(args.DUIF_ckpt)
    DUIF_model.eval()
    with torch.no_grad():
        # construct topN rec graph
        rec_graph = torch.zeros((model.num_user, model.num_item)) #sp.dok_matrix((model.num_user, model.num_item), dtype=np.float32)
        for u_id in range(topN_rank.size(0)):
            for i_id in topN_rank[u_id]:
                rec_graph[u_id,i_id] = 1
        with torch.no_grad():
            DUIF_scores = rescore(DUIF_model, lowconf_mask, torch.tensor(train_dataset.ui_graph.toarray()), rec_graph)
            final_scores = (1-lowconf_mask) * all_scores_org + lowconf_mask * (all_scores_org + DUIF_scores) / 2
        print('saving scores...')
        np.save(f'final_scores_{args.backmodel}',final_scores)
        print('scores saved.')
        
        test_result = rerank(test_data, final_scores, model.num_user, args.step, topK, None)
        test_result_warm = rerank(test_warm_data, final_scores, model.num_user, args.step, topK, cold_item)
        test_result_cold = rerank(test_cold_data, final_scores, model.num_user, args.step, topK, warm_item)

        print(f'--- Double Check Performance ---')
        print('==='*18)
        print('All')
        print_results(None,None,test_result)
        print('Warm')
        print_results(None,None,test_result_warm)
        print('Cold')
        print_results(None,None,test_result_cold)
        print('==='*18)
    print('End.')