#-*- coding: utf-8 -*-

import time
from collections import defaultdict

import numpy as np
import os
import torch
from torch import nn
from tqdm import tqdm
from utils import colorize
import pandas as pd

import pickle 

from train.base_trainer import BaseTrainer

class Trainer(BaseTrainer):

    def train(self, dataloader, mode='train', epoch=-1):

        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        self.dataloader = dataloader

        tb_stepmod = (100 if mode == 'train' else 1) if not self.opts.test_only else 1

        bs = self.opts.batch_size
        counter = 0

        # cummulative losses and metrics dictionary
        metrics_dict = defaultdict( float)  

        bar = tqdm(total=len(dataloader))

        self.data_tic = self.step_tic = None
                
        if mode == 'test': 
            test_queries = pd.read_csv('test_set_queries.csv')
            dict_test_out = {}
            for vid in set(test_queries.videos.tolist()):
                dict_test_out[vid] = {'feats_idx': [], 'topk': []}
        
        for b_id, batch_sample in enumerate(dataloader): 

            self.model.zero_grad()
            model_out = self.model.forward(batch_sample)
            # import pdb; pdb.set_trace()

            ### in text mode,  evaluate query data
            if mode == 'test':

                for i in range(len(batch_sample['feats_idx'])): 
                    dict_test_out[batch_sample['video'][i]]['feats_idx'].append(int(batch_sample['feats_idx'][i].numpy()))
                    dict_test_out[batch_sample['video'][i]]['topk'].append(model_out['topk'][i].numpy())

            # ------------------------- Time steps  -------------------------

            if self.step_tic:
                metrics_dict['t'] += time.time() - self.step_tic
                if self.data_tic is not None:
                    metrics_dict['dt'] += time.time() - self.data_tic
                    metrics_dict['dt/total'] = (
                        metrics_dict['dt'] / metrics_dict['t']) * (counter + 1)
            self.step_tic = time.time()

            # ------------------------- Loss  -------------------------
            loss = model_out['loss'].mean()

            # ------------------------- Backprop  -------------------------
            if mode == 'train':
                loss.backward(retain_graph=False)

                if self.opts.grad_clip_norm:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(),
                                                self.opts.grad_clip_norm)

                self.optimizer.step()

            # ------------------------- Metrics  -------------------------

            metrics_dict['loss'] += model_out['loss'].mean().detach().cpu().item()

            # - tb summaries
            if (self.opts.test_only or
                    mode == 'train' and ( b_id % tb_stepmod) == 0
                        or b_id == len(dataloader) - 1):
                for loss_name, loss_val in metrics_dict.items():
                    self.tb_writer.add_scalar(f'{mode}/{loss_name}',
                                            loss_val / (counter + 1),
                                            self.global_step)

            counter += 1
            if mode == 'train' or self.opts.test_only:
                self.global_step += 1

            bar.update(1)

            desc = "%s: " % mode
            for cuml_name, cuml in sorted(metrics_dict.items()):
                desc += "%s %.2f " % (cuml_name, cuml / counter)
            bar.set_description(desc)

            self.data_tic = time.time(
            )  # this counts how long we are waiting for data

        bar.close()

        if mode=='test':
            print('Evaluating test set queries...')
            detected = []
            start_end_idx = []
            for qr in tqdm(range(len(test_queries))): 
                vid = test_queries.loc[qr].videos
                word = test_queries.loc[qr].queries
                topk_pred = np.array([dict_test_out[vid]['topk'][j] for j in dict_test_out[vid]['feats_idx']])
                if word in topk_pred: 
                    detected.append(1)
                    for row in range(len(topk_pred)): 
                        if word in topk_pred[row, :]: 
                            start = row
                            break
                    for row in reversed(range(len(topk_pred))): 
                        if word in topk_pred[row, :]: 
                            end = row
                            break
                    start_end = [start, end]
                else: 
                    detected.append(0)
                    start_end = [-1, -1]
    
                start_end_idx.append(start_end)

            starts = [s[0] for s in start_end_idx]
            ends = [s[1] for s in start_end_idx]

            dict_out = pd.DataFrame({'detections': detected, 'start': starts, 'end': ends})
            dict_out.to_csv(os.path.join('submission.csv'), header=False)
            print('Saved ', os.path.join('submission.csv'))
    
        desc = "Epoch end: %s: " % mode
        for cuml_name, cuml in sorted(metrics_dict.items()):
            desc += "%s %.2f " % (cuml_name, cuml / counter)
            self.tb_writer.add_scalar(f'{mode}_epoch/{cuml_name}',
                                      cuml / counter, self.global_step)
        print(desc)
        self.tb_writer.flush()

        return bar.desc 


