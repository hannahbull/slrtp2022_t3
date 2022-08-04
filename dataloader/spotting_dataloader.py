import os
import pickle
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scipy import io
import json 

import torch

class SpottingDataset(Dataset): 
    
    def __init__(self, mode, opts):
        self.mode = mode
        self.opts = opts

        self.vocab_file = json.load(open(self.opts.vocab_file_loc, 'r'))

        ### Build data dictionary by iterating through vtt files and captions
        self.data_dict = {}
        self.data_dict["txt"] = []
        self.data_dict["txt_idx"] = []
        self.data_dict["feats"] = []
        self.data_dict["feats_idx"] = []
        self.data_dict["video"] = []

        ### Load list of train/val/test split videos
        if (self.mode == 'train'): 
            data_path_root = self.opts.train_data_loc
            labels_path_root = self.opts.train_labels_loc
        elif (self.mode == 'val'): 
            data_path_root = self.opts.val_data_loc
            labels_path_root = self.opts.val_labels_loc
        elif (self.mode == 'test'):
            data_path_root = self.opts.test_data_loc
        else: 
            print('Choose mode = "train" or "val" or "test"')
        
        data_paths = os.listdir(data_path_root)
        if self.opts.random_subset_data < len(data_paths): 
            random.seed(123)
            # data_paths = random.sample(data_paths, self.opts.random_subset_data)
            data_paths = data_paths[0:self.opts.random_subset_data]
            print('Number of videos ', len(data_paths), data_paths)

        print('Loading data... ')
        for dp in tqdm(data_paths): 
            load_feats = np.load(os.path.join(data_path_root, dp))
            pad_start = np.tile(load_feats[0,:], 8).reshape(8,-1)
            pad_end = np.tile(load_feats[-1,:], 7).reshape(7,-1)
            load_feats = np.concatenate((pad_start, load_feats, pad_end))
            if self.mode != 'test':
                load_labels = np.load(os.path.join(labels_path_root, dp))
            else: 
                load_labels = np.ones(len(load_feats))*980
            
            for ix, lab in enumerate(load_labels): 
                if lab < len(self.vocab_file) and lab >=0:
                    self.data_dict["txt"].append(self.vocab_file[str(int(lab))])
                    self.data_dict["txt_idx"].append(lab)
                else: 
                    print('Warning, OOV word?')
                    # import pdb; pdb.set_trace()
                    self.data_dict["txt"].append("SILENCE")
                    self.data_dict["txt_idx"].append(980)
                self.data_dict["feats"].append(load_feats[ix,:])
                self.data_dict["feats_idx"].append(ix)
                self.data_dict["video"].append(dp)

    def __len__(self): 
        return len(self.data_dict["txt"])

    def __getitem__(self, index):

        out_dict = {}
        out_dict["video"] = self.data_dict["video"][index]
        out_dict["txt_idx"] = self.data_dict["txt_idx"][index]
        out_dict["txt"] = self.data_dict["txt"][index]
        out_dict["feats"] = self.data_dict["feats"][index]
        out_dict["feats_idx"] = self.data_dict["feats_idx"][index]

        return out_dict
