'''
Training code for MRBrainS18 datasets segmentation
Written by Whalechen
'''

from setting import parse_opts
from datasets.DRF_data import DRF_data
from DRF_model import generate_model
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time
from utils.logger import log
from scipy import ndimage
import os
import torch.backends.cudnn as cudnn
import csv 
import logging

def get_lr(optimzer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() 
    acc = (correct / len(y_pred)) * 100 
    return acc
    
def train(data_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets):
    # settings
    
    logging.basicConfig(filename=results_file, mode = 'a', format='%(asctime)s - %(message)s', level = logging.INFO, force = True)
    
    batches_per_epoch = len(data_loader)
    #print("batches per epoch is" + str(batches_per_epoch))
    #log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    logger.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    loss_clas = nn.BCELoss()
    
    # define log file for results
    print(results_file)
    print("Current setting is:")
    print(sets)
    print("\n\n")
    if not sets.no_cuda:
        loss_clas = loss_clas.cuda()
        print("cuda is used")

    model.train()
    train_time_sp = time.time()
 
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))
        
        optimizer.step()
        #scheduler.step()
        #log.info('lr = {}'.format(scheduler.get_lr()))
        #logger.info('Epoch = {}, lr = {}'.format(epoch, scheduler.get_lr()))
        
        log.info('lr = {}'.format(get_lr(optimizer)))
        logger.info('Epoch, {}, lr, {}'.format(epoch, get_lr(optimizer)))
        
        correct = 0
        total = 0
        epoch_loss = 0
        batch_loss = 0
    
        for batch_id, batch_data in enumerate(data_loader):
            
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes, labels = batch_data
            
            if not sets.no_cuda:
                volumes = volumes.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            #Posterior probabilites 
            raw_output = model(volumes)
            output = torch.round(raw_output)
            
            #Calculate loss
            loss_value_clas = loss_clas(raw_output.squeeze(), labels.squeeze())

            
            #Save acc, loss and post prob 
            correct += torch.eq(labels.squeeze(), output.squeeze()).sum().item() 
            batch_loss += loss_value_clas.item() * sets.batch_size
            total += labels.size(0)

            #loss + backward
            loss = loss_value_clas
            loss.backward()
            
            # Update
            optimizer.step()
            #scheduler.step()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                'Batch: {}-{} ({}), loss = {:.3f}, avg_batch_time = {:.3f}' \
                    .format(epoch, batch_id, batch_id_sp, loss.item(),  avg_batch_time))
                  
            if not sets.ci_test:
                # save model
                if batch_id == 0 and batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                    # if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                    model_save_path = '{}_epoch_{}.pth.tar'.format(save_folder, epoch)
                    model_save_dir = os.path.dirname(model_save_path)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)
                #always save last peoch
 