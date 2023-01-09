import torch
import torch.nn as nn
from tqdm import tqdm
import random

def train(epoch, length, dataloader, model, optimizer, batch_size, align_all=False, writer=None):    
    model.train()
    sum_loss = 0.0
    sum_reg_loss = 0.0
    step = 0.0
    num_pbar = 0
    sum_mat = 0.0

    for user_tensor, item_tensor, aug_negs in dataloader:
        align = 0 if align_all else random.randint(1,3)
        
        optimizer.zero_grad()
        loss, reg_loss = model.loss(user_tensor.cuda(), item_tensor.cuda(), aug_negs.cuda(), align)
        loss.backward()
        optimizer.step()
        sum_loss += loss.cpu().item()
        sum_reg_loss += reg_loss.cpu().item()
        num_pbar += batch_size
        step += 1.0

    return loss, sum_mat/step
