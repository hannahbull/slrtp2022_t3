from pickle import load
from config.config import load_opts, save_opts

import os
opts=load_opts()
os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id

from torch.utils.data import DataLoader
from torch import nn

import os 

import torch

import time

from train import trainer_dict
from dataloader import dataset_dict
from models import model_dict

import numpy as np

import random 

from pathlib import Path
import glob

from multiprocessing import Pool

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(opts):

    print('Cuda current device ', torch.cuda.current_device())

    set_seed(42)

    if not opts.test_only:

        def worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id+int(time.time()))

        dataset = dataset_dict[opts.dataset](mode='train', opts=opts)
        dataloader = DataLoader(
            dataset,
            batch_size=opts.batch_size,
            shuffle=False,
            num_workers=opts.n_workers,
            worker_init_fn=worker_init_fn,
        )

        dataset_val = dataset_dict[opts.dataset](mode='val', opts=opts)
        dataloader_val = DataLoader(dataset_val,
                                    batch_size=opts.batch_size,
                                    shuffle=False,
                                    num_workers=opts.n_workers)

    else:
        dataset = dataset_dict[opts.dataset](mode='test', opts=opts)
        dataloader = DataLoader(dataset,
                                    batch_size=opts.batch_size,
                                    shuffle=False,
                                    num_workers=opts.n_workers)


    if len(dataloader)>0: 
        model = model_dict[opts.model](opts=opts, dataloader=dataloader)
        print("Model's state_dict:")

        trainer = trainer_dict[opts.trainer](model, opts)

        if opts.resume:
            trainer.load_checkpoint(opts.resume)

        if not opts.test_only:
            save_opts(opts, opts.save_path + "/args.txt")
        
        if not opts.test_only:
            scorefile = open(opts.save_path + "/scores.txt", "a+")

            res_val = trainer.train(
                dataloader_val, mode='val', epoch=-1)  # initialize metric

            for epoch in range(opts.n_epochs):
                print('Epoch {:d}/{:d}'.format(epoch, opts.n_epochs))
                
                res_tr = trainer.train(dataloader,
                                        mode='train',
                                        epoch=epoch)

                # -- evaluate
                with torch.no_grad():  # to save memory
                    res_val = trainer.train(dataloader_val,
                                                        mode='val',
                                                        epoch=epoch)

                    scorefile.write("{} | {}\n".format(res_tr, res_val))
                    scorefile.flush()

                    print('saving model ', "model_{:010d}.pt".format(trainer.global_step))
                    model_ckpt = "model_{:010d}.pt".format(trainer.global_step)
                    trainer.save_checkpoint(model_ckpt)
                        

            scorefile.close()
        else:
            trainer.train(dataloader,
                                mode='test',
                                epoch=0)

    else:
        print('Length of dataloader is 0') 

if __name__ == '__main__':
    opts = load_opts()     
    main(opts)
