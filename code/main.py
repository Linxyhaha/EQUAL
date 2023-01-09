import argparse
import os
import time
from xml.etree.ElementInclude import default_loader
import numpy as np
import torch
import random
from Dataset import TrainingDataset, data_load
from model_CLCRec import CLCRec
from torch.utils.data import DataLoader
from Train import train
from Full_rank import full_ranking, sub_ranking
from Metric import print_results
import ipdb
import pdb

###############################248###########################################

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Seed init.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--data_path', default='amazon_clothing', help='Dataset path')
    parser.add_argument('--save_file', default='', help='Filename')

    parser.add_argument('--PATH_weight_load', default=None, help='Loading weight filename.')
    parser.add_argument('--PATH_weight_save', default=None, help='Writing weight filename.')
    parser.add_argument('--prefix', default='', help='Prefix of save_file.')
    
    # CLCRec
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
    parser.add_argument('--log_name', type=str, default='log', help='log name')
    parser.add_argument('--inference',action='store_true', help='only inference stage')
    parser.add_argument('--ckpt', type=str, help='pretrained model path')

    # ET
    parser.add_argument('--alpha', type=float, default=0.1, help='hyper parameter for beta distribution sampling')
    parser.add_argument('--pos_ratio', type=float, default=0.2, help='ratio for positive items')
    parser.add_argument('--neg_ratio', type=float, default=0.2, help='ratio for negative items')

    parser.add_argument('--align_all', type=int, default=0, help='whether or not apply all the alignment in one batch')
    parser.add_argument('--mse_weight', type=float, default=0.001, help='mse loss weight for alignment between raw feature and feature rep')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = init()
    
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
    alpha = args.alpha
    pos_ratio = args.pos_ratio
    neg_ratio = args.neg_ratio
    mse_weight = args.mse_weight
    align_all = args.align_all

    has_v = True if args.has_v == 'True' else False
    has_a = True if args.has_a == 'True' else False
    has_t = True if args.has_t == 'True' else False

    dim_E = args.dim_E
    is_word = True if data_path == 'tiktok' else False
    writer = None

    ##########################################################################################################################################
    print('Data loading ...')

    num_user, num_item, num_warm_item, pre_user_embedding, pre_item_embedding, train_data, val_data, val_warm_data, val_cold_data, test_data, test_warm_data, test_cold_data, v_feat, a_feat, t_feat = data_load(data_path)
    
    dir_str = f'../data/{args.data_path}'
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

    train_dataset = TrainingDataset(num_user, num_item, user_item_all_dict, cold_item, train_data, num_neg, pre_user_embedding, pre_item_embedding, pos_ratio, neg_ratio)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)

    print('Data has been loaded.')
    ##########################################################################################################################################
    # inference stage
    if args.inference:
        with torch.no_grad():
            model = torch.load('models/' + args.ckpt)
            #val_result = full_ranking(0,  model, val_data, train_data, None, False, step, topK, 'Val/', writer)
            test_result = full_ranking(0, model, test_data, tv_dict, None, False, step, topK, 'Test/', writer)
            test_result_warm = full_ranking(0, model, test_warm_data, tv_dict, cold_item, False, step, topK, 'Test/warm_', writer)
            test_result_cold = full_ranking(0, model, test_cold_data, tv_dict, warm_item, False, step, topK, 'Test/cold_', writer)
            print('---'*18)
            print('All')
            print_results(None,None,test_result)
            print('Warm')
            print_results(None,None,test_result_warm)
            print('Cold')
            print_results(None,None,test_result_cold)
        os._exit(1)
    ##########################################################################################################################################
    # Build Model
    model = CLCRec(warm_item, cold_item, num_user, num_item, num_warm_item, train_data, reg_weight, dim_E, v_feat, a_feat, t_feat, temp_value, num_neg, lr_lambda, is_word, alpha, mse_weight, num_sample).cuda()
    print('Model has been loaded.')
    ##########################################################################################################################################
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rate}])#, 'weight_decay': reg_weight}])
    ##########################################################################################################################################
    max_precision = 0.0
    max_recall = 0.0
    max_NDCG = 0.0
    num_decreases = 0 
    best_epoch = 0
    max_val_result = max_val_result_warm = max_val_result_cold = max_test_result = max_test_result_warm = max_test_result_cold = None
    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        loss, mat = train(epoch, len(train_dataset), train_dataloader, model, optimizer, batch_size, args.align_all, writer)
        elapsed_time = time.time() - epoch_start_time
        print("Train: The time elapse of epoch {:03d}".format(epoch) + " is: " + 
                time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        if torch.isnan(loss):
            print('loss is nan. Quit')
            break
        torch.cuda.empty_cache()
        if (epoch+1)%5==0:   
            print('start test') 
            test_result = None
            
            with torch.no_grad():
                val_result = full_ranking(0, model, val_data, train_dict, None, False, step, topK, 'Test/', writer)
                test_result = full_ranking(0, model, test_data, tv_dict, None, False, step, topK, 'Test/', writer)
                test_result_warm = full_ranking(0, model, test_warm_data, tv_dict, cold_item, False, step, topK, 'Test/warm_', writer)
                test_result_cold = full_ranking(0, model, test_cold_data, tv_dict, warm_item, False, step, topK, 'Test/cold_', writer)

            print('---'*18)
            print("Runing Epoch {:03d} ".format(epoch) + " costs " + time.strftime(
                                "%H: %M: %S", time.gmtime(time.time()-epoch_start_time)))
            print_results(None,val_result,test_result)
            print('All')
            print_results(None,val_result,test_result)
            print('Warm')
            print_results(None,None,test_result_warm)
            print('Cold')
            print_results(None,None,test_result_cold)
            print('---'*18)

            if val_result[1][0] > max_recall:
                best_epoch=epoch
                pre_id_embedding = model.id_embedding
                max_recall = val_result[1][0]
                max_val_result = val_result
                max_test_result = test_result
                max_test_result_warm = test_result_warm
                max_test_result_cold = test_result_cold
                num_decreases = 0
                if not os.path.exists(args.save_path):
                    os.mkdir(args.save_path)
                torch.save(model, '{}{}_{}bs_{}lr_{}reg_{}ng_{}lam_{}rou_{}temp_{}dimE_{}alpha_{}pos_{}neg_{}mse_{}all_{}.pth'.format(args.save_path, \
                                    args.model_name, args.batch_size, args.l_r, args.reg_weight, args.num_neg, args.lr_lambda, \
                                    args.num_sample, args.temp_value, args.dim_E, args.alpha, args.pos_ratio, args.neg_ratio, \
                                    args.mse_weight, args.align_all, args.log_name))
            else:
                if num_decreases > 5:
                    print('-'*18)
                    print('Exiting from training early')
                    break
                else:
                    num_decreases += 1

    print('==='*18)
    print(f"End. Best Epoch is {best_epoch}")
    print('---'*18)
    print('All')
    print_results(None, None, max_test_result)
    print('Warm')
    print_results(None,None,max_test_result_warm)
    print('Cold')
    print_results(None,None,max_test_result_cold)
    print('---'*18)