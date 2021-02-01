#
# Author : Alwyn Mathew
# Modifier : KuoShiuan Peng
#
# Monodepth in pytorch(https://github.com/alwynmathew/monodepth-pytorch)
#

import time
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data.dataloader import StereoDataloader
import matplotlib.pyplot as plt

from network.depth_modelv2 import Model
from utils.visualizer import Visualizer
#from torch.multiprocessing import Pool, Process, set_start_method

class Train:

    def __init__(self, opt):
        self.params = opt

    def train(self):
        params = self.params        
        # loading data
        # loading training images

        #set_start_method('spawn')

        data_loader = StereoDataloader(params) # create dataloader 
        train_data = DataLoader(data_loader, batch_size=params.batchsize, shuffle=True)#, num_workers=1)
        dataset_size = len(data_loader)
        print('#training images: %d' %dataset_size)

        start_epoch, epoch_iter = 1, 0
        total_steps = (start_epoch - 1) * dataset_size + epoch_iter
        iter_ = 1

        # create/load model
        model = Model(params)
        visualizer = Visualizer(params)

        total_train_size = dataset_size*params.niter

        print('>>> dataset_size: %d, total train size: %d' %(dataset_size, total_train_size))
        print('\nTraining started...')
        train_start_time = time.time()
        # for epoch in range(start_epoch, params.niter + params.niter_decay + 1):
        for epoch in range(start_epoch, params.niter+1):
            # epoch start time
            epoch_start_time = time.time()
            # params.current = epoch

            for i, data in enumerate(train_data, start=epoch_iter):
                iter_start_time = time.time()
                total_steps += params.batchsize
                epoch_iter += params.batchsize

                # whether to collect output images
                params.save_fake = total_steps % params.display_freq == 0

                if params.save_fake:
                    time_spent = time.time()-train_start_time

                    left_dataize = total_train_size - total_steps
                    fps = round(total_steps*1.0/time_spent,1)
                    time_left = round(left_dataize / fps / 3600,2)

                    print('\nEpoch: %d, Iteration: %d, Time Left: %2.2f hrs, examples/s: %3.2f' \
                        %(epoch, total_steps, time_left, fps))

                # when to start dis traning
                if epoch > params.headstart: 
                        params.headstart_switch = -1

                
                # forward
                loss_G = model(Variable(data['left_img']), Variable(data['right_img']))

                # backward G
                model.optimizer_G.zero_grad()
                loss_G.backward()
                model.optimizer_G.step()

                # # display input & output and save ouput images
                if params.save_fake:
                    result_img = model.get_result_img(Variable(data['left_img']), Variable(data['right_img']))
                    visualizer.display_current_results(result_img, epoch, total_steps)

            # epoch end time
            iter_end_time = time.time()
            print('End of epoch %d / %d \t Time Taken: %d sec' %(epoch, params.niter, time.time() - epoch_start_time))


            # save mdodel
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save(epoch)