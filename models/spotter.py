import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import math


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Spotter(nn.Module):
    """
    Spotter MLP model
    """

    def __init__(self, opts, dataloader):
        super().__init__()

        self.opts = opts
        self.dataloader = dataloader

        self.fc = MLP(input_dim=1024, hidden_dim=256, output_dim=981, num_layers=3)

        self.softmax = nn.Softmax()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, data_dict):

        out = {}
        
        feat_inp = data_dict['feats'].type(torch.FloatTensor).cuda() 
                
        target_class = data_dict['txt_idx'].type(torch.LongTensor).cuda()

        if not self.opts.logits_only:
            lin_layer = self.fc(feat_inp)
        else: 
            lin_layer = feat_inp

        loss = self.loss(lin_layer, target_class)

        out['loss'] = loss
        out['topk'] = torch.topk(lin_layer.detach(), 5)[1].cpu()
        return out
