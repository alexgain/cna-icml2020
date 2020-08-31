import torch
import numpy as np
import matplotlib.pyplot as plt

from cna_utils import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--net_path', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--flatten', type=int, default=0)
parser.add_argument('--cuda', type=int, default=0)

args = parser.parse_args()

if not args.cuda:
    net = torch.load(args.net_path, map_location=torch.device('cpu'))
else:
    net = torch.load(args.net_path)

train_loader, test_loader = get_dataset_loaders(args.dataset)
cna_val = CNA_all(train_loader, net, args.flatten)
cna_m_val = CNA_M(train_loader, test_loader, net, args.flatten)

print("CNA value:",cna_val)
print("CNA-Margin value:",cna_val)