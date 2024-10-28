import os
import copy
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from torch import optim
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split

desired_number_threads = 5
torch.set_num_threads(desired_number_threads)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for processing')
    parser.add_argument('--dataname', type=str, default='buchwald', help='Dataset argument')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension argument')
    parser.add_argument('--lr', type=float, default=0.001, help='Optimizer kwargs argument')
    parser.add_argument('--epochs', type=int, default=300, help='Epochs argument')
    parser.add_argument("--weight_decay", type=float, default = 0)
    parser.add_argument('--repeat', type=int, default=10, help='Seed argument')
    parser.add_argument('--device', type=int, default=0, help='Device to use for processing')
    parser.add_argument("--split_type", type=str, default='random')
    parser.add_argument("--label_name", type=str, default='label')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = parse_arguments()
    device = args.device

    data = ReactionDataset(dataname)
    data.load_data()

    for random_state in range(repeat):

        torch.manual_seed(random_state)
        train, valid = train_test_split(data.processed, train_size=0.6, random_state=random_state)
        valid, test = train_test_split(valid, train_size=0.5, random_state=random_state)

        train_loader = DataLoader(train, batch_size = args.batch_size, shuffle=True)
        test_loader = DataLoader(test, batch_size = args.batch_size)
        valid_loader = DataLoader(valid, batch_size = args.batch_size)

        trainer = Trainer(args, train_loader, valid_loader, test_loader, random_state)
        trainer.train()









