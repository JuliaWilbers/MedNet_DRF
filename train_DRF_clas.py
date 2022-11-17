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
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
import pandas as pd


def get_lr(optimzer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def train(data_loader, validation_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets):
    
    # settings
    batches_per_epoch = len(data_loader)
    loss_clas = nn.BCELoss()
    print("Current setting is:")
    print(sets)
    print("\n\n")
    if not sets.no_cuda:
        loss_clas = loss_clas.cuda()

    train_time_sp = time.time()
    epoch_l, val_l, train_l, tacc_l, lacc_l = ([] for i in range(5))
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))
        
        optimizer.step()
        #log.info('lr = {}'.format(scheduler.get_lr()))

        log.info('lr = {}'.format(get_lr(optimizer)))
        
        # Parameter init
        correct, total, tloss, running_loss, vloss, vcorrect, running_vloss, vtotal = (0 for i in range(8))
        epoch_l.append(epoch)
        
        # Training 
        model.train()
        for batch_id, batch_data in enumerate(data_loader):

            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes, masks, labels = batch_data
            if not sets.no_cuda:
                volumes = volumes.cuda()
                labels = labels.cuda()
                masks = masks.cuda()
                
            optimizer.zero_grad()

            # Posterior probabilites
            raw_output = model(volumes, masks)
            output = torch.round(raw_output)

            # calculate loss
            loss = loss_clas(raw_output.squeeze(), labels.squeeze())

            # calculate batch loss,
            correct += torch.eq(labels.squeeze(), output.squeeze()).sum().item()
            running_loss += loss.item()
            total += labels.size(0)

            # loss + backward
            loss.backward()
            optimizer.step()
            #scheduler.step()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                'Batch: {}-{}, loss = {:.3f}' \
                    .format(epoch, batch_id, running_loss))

        # Calculate and log epoch loss and accuracy
        epoch_acc = 100. * correct / total
        tloss = running_loss / len(data_loader)
        train_l.append(tloss)
        tacc_l.append(epoch_acc)
        log.info('Epoch = {}, Epoch_Loss = {}, Accuracy = {}'.format(epoch, tloss, epoch_acc))
        
        # Validation
        model.eval()
        with torch.no_grad():  
          for batch_id, batch_data in enumerate(validation_loader):
              # get validation data
              volumes, masks, labels = batch_data
              if not sets.no_cuda:
                  volumes = volumes.cuda()
                  labels = labels.cuda()
                  masks = masks.cuda()
              # forward    
              y_pb = model(volumes, masks)
              output = torch.round(y_pb)
              
              vloss = loss_clas(y_pb.squeeze(), labels.squeeze())
              vcorrect += torch.eq(labels.squeeze(), output.squeeze()).sum().item()
              running_vloss += vloss.item() 
              vtotal += labels.size(0)
              
          # Calculate and log validation loss and accuracy
          vacc = 100. * vcorrect / vtotal
          vloss = running_vloss / len(validation_loader)
        
          val_l.append(vloss)
          log.info('Epoch = {}, validation_Loss = {}, Accuracy validation= {}'.format(epoch, vloss, vacc))
          lacc_l.append(vacc)
          scheduler.step(vloss)
          if epoch == 0:
            val_l.append(tloss)
          else:
            val_l.append(vloss)
        
        # Save model in interval
        if sets.save_trails == True and epoch %10 == sets.save_intervals:
          model_save_path = './trails/DRF_models/{}_set_{}_int_ep{}.pth.tar'.format(sets.method, sets.setnr, epoch)
          log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id))
          torch.save({
          'epoch': epoch,
          'batch_id': batch_id,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict()},
          model_save_path)
              
        # Save best / last model (model with lowest validation loss)   
        if epoch == 0:
            best_val_loss  = vloss
                
        if vloss < best_val_loss:
          best_val_loss = vloss
          model_save_path = './trails/DRF_models/{}_set_{}best.pth.tar'.format(sets.method, sets.setnr)
          log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id))
          torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()}, 
            model_save_path)
            
        elif epoch == (total_epochs-1):
          model_save_path = './trails/DRF_models/{}_set_{}last.pth.tar'.format(sets.method, sets.setnr)
          log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id))
          torch.save({
          'epoch': epoch,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict()}, 
          model_save_path)
          

    print('Finished training')
    
    # Store results
    df = pd.DataFrame(list(zip(*[epoch_l, train_l, val_l, tacc_l, lacc_l])), columns = ['Epoch', 'training_loss', 'validation_loss', 'training acc', 'validation acc'])
    df.to_csv(sets.results_file, index=False)
    
    # Store learning curve
    plt.plot(np.array(epoch_l), np.array(train_l))
    plt.plot(np.array(epoch_l), np.array(val_l))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve Experiment {}".format(sets.method))
    plt.savefig('./results/{}_set_{}_lc.png'.format(sets.method, sets.setnr))
    
    if sets.ci_test:
        exit()


if __name__ == '__main__':
    # settting
    sets = parse_opts()
    sets.no_cuda = False
    sets.model_depth = 10
    sets.resnet_shortcut = 'A'
    sets.input_D = 210  # Z
    sets.input_H = 140  # Y
    sets.input_W = 150  # X
    sets.learning_rate = 0.0001
    sets.data_dir = './data/DRF_data/'
    

    # Check /change
    if sets.augmentation == 'True':
        sets.label_list = './data/DRF_label_augmented_sets/set_{}/train_new_augmented.txt'.format(sets.setnr)
        sets.label_list_val = './data/DRF_label_sets/set_{}/validation.txt'.format(sets.setnr)
        sets.method = 'method{}_a_v{}'.format(sets.methodnr, sets.version)
    else:
        sets.label_list = './data/DRF_label_sets/set_{}/train_new.txt'.format(sets.setnr)
        sets.label_list_val = './data/DRF_label_sets/set_{}/validation.txt'.format(sets.setnr)
        sets.method = 'method{}_v{}'.format(sets.methodnr, sets.version)

    # Check / change method 2
    if sets.methodnr == '2':
        sets.pretrained = False
    else:
        sets.pretrain_path = './pretrain/resnet_10.pth'
        sets.pretrained = True

    # Set name of the results file
    sets.results_file = "results/{}_{}_set{}.csv".format(sets.method, sets.phase, sets.setnr)

    # getting mode
    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets)
    #params = [{'params': parameters, 'lr': sets.learning_rate}]

    # optimizer

    if sets.pretrained:
        params = [
            {'params': parameters['base_parameters'], 'lr': 0},
            {'params': parameters['new_parameters'], 'lr': sets.learning_rate}
            ]

        # {'params': parameters['new_parameters'], 'lr': sets.learning_rate * 100}]
    else:
        params = [{'params': parameters, 'lr': sets.learning_rate}]

    #optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)
    optimizer = torch.optim.Adam(params, sets.learning_rate)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma = 1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-8, eps=1e-08)
    #scheduler = None
    log.info(
        'Learning rate: {}, Optimizer: {}, Scheduler: {}'.format(sets.learning_rate, optimizer.__class__.__name__,
                                                                 scheduler.__class__.__name__))

    # train from resume
    if sets.resume_path:
        if os.path.isfile(sets.resume_path):
            print("=> loading checkpoint '{}'".format(sets.resume_path))
            checkpoint = torch.load(sets.resume_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(sets.resume_path, checkpoint['epoch']))

    # getting data
    sets.phase = 'train'
    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True

    training_dataset = DRF_data(sets.data_dir, sets.label_list, sets)
    validation_dataset = DRF_data(sets.data_dir, sets.label_list_val, sets)
    
    data_loader = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers,
                             pin_memory=sets.pin_memory)
    
    validation_loader = DataLoader(validation_dataset, batch_size=sets.batch_size, shuffle=False, num_workers=sets.num_workers,
                             pin_memory=sets.pin_memory)
    # training
    train(data_loader, validation_loader, model, optimizer, scheduler, total_epochs=sets.n_epochs, save_interval=sets.save_intervals,
          save_folder=sets.save_folder, sets=sets)
    