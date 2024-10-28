import torch
import numpy as np

def create_batch_mask(samples):

    batch0 = samples[0].batch.reshape(1, -1) 
    index0 = torch.cat([batch0, torch.tensor(range(batch0.shape[1])).reshape(1, -1)])
    mask0 = torch.sparse_coo_tensor(index0, torch.ones(index0.shape[1]), size = (batch0.max() + 1, batch0.shape[1]))

    batch1 = samples[1].batch.reshape(1, -1)
    index1 = torch.cat([batch1, torch.tensor(range(batch1.shape[1])).reshape(1, -1)])
    mask1 = torch.sparse_coo_tensor(index1, torch.ones(index1.shape[1]), size = (batch1.max() + 1, batch1.shape[1]))

    return mask0, mask1

def r2_score(y_pred, y_true):
    y_true_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_true_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_residual / ss_total
    return r2


def MC_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    pass
