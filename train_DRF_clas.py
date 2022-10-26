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


def train(data_loader, validation_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets):
    # settings

    logging.basicConfig(filename=results_file, mode='a', format='%(asctime)s - %(message)s', level=logging.INFO,
                        force=True)

    batches_per_epoch = len(data_loader)
    # print("batches per epoch is" + str(batches_per_epoch))
    # log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
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

    train_time_sp = time.time()

    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))

        optimizer.step()
        # scheduler.step()
        # log.info('lr = {}'.format(scheduler.get_lr()))
        # logger.info('Epoch = {}, lr = {}'.format(epoch, scheduler.get_lr()))

        log.info('lr = {}'.format(get_lr(optimizer)))
        logger.info('Epoch = {}, lr = {}'.format(epoch, get_lr(optimizer)))
        
        # Parameter init
        correct = 0
        total = 0
        epoch_loss = 0
        batch_loss = 0
        
        val_loss = 0
        correct_val = 0
        batch_loss_val = 0
        total_val = 0
        
        model.train()
        
        # Training 
        for batch_id, batch_data in enumerate(data_loader):

            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes, labels = batch_data
            if not sets.no_cuda:
                volumes = volumes.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            # Posterior probabilites
            raw_output = model(volumes)
            output = torch.round(raw_output)

            # calculate acc and loss
            loss_value_clas = loss_clas(raw_output.squeeze(), labels.squeeze())

            # Save acc, loss and post prob
            correct += torch.eq(labels.squeeze(), output.squeeze()).sum().item()
            batch_loss += loss_value_clas.item() * sets.batch_size
            total += labels.size(0)

            # loss + backward
            loss = loss_value_clas
            loss.backward()
            optimizer.step()
            # scheduler.step()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                'Batch: {}-{} ({}), loss = {:.3f},  avg_batch_time = {:.3f}' \
                    .format(epoch, batch_id, batch_id_sp, loss.item(), avg_batch_time))

            if not sets.ci_test:
                # save model
                if batch_id == 0 and sets.save_trails == True and batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                    # if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                    model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(save_folder, epoch, batch_id)
                    model_save_dir = os.path.dirname(model_save_path)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)

                if epoch == (sets.n_epochs - 1):
                    model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(save_folder, epoch, batch_id)
                    model_save_dir = os.path.dirname(model_save_path)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)

                    log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id))
                    torch.save({
                        'epoch': epoch,
                        'batch_id': batch_id,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        model_save_path)

        acc_total = 100. * correct / total
        epoch_loss = batch_loss / len(data_loader)
        logger.info('Epoch = {}, Epoch_Loss = {}, Accuracy = {}'.format(epoch, epoch_loss, acc_total))
        
        # Validation
        model.eval() 
        for batch_id, batch_data in enumerate(validation_loader):
        # forward
            volumes, labels = batch_data
            if not sets.no_cuda:
                volumes = volumes.cuda()
                labels = labels.cuda()
            
            with torch.no_grad():
                y_pb = model(volumes)
                output = torch.round(y_pb)
            
                val_loss = loss_clas(y_pb.squeeze(), labels.squeeze())
                correct_val += torch.eq(labels.squeeze(), output.squeeze()).sum().item()
                batch_loss_val += loss_value_clas.item() * sets.batch_size
                total_val += labels.size(0)
        
        val_acc = 100. * correct_val / total_val
        val_loss = batch_loss_val / len(validation_loader)
        logger.info('Epoch = {}, validation_Loss = {}, Accuracy validation= {}'.format(epoch, val_loss, val_acc))

    print('Finished training')
    if sets.ci_test:
        exit()


if __name__ == '__main__':
    # settting
    sets = parse_opts()
    sets.n_epochs = 100
    sets.no_cuda = False
    sets.num_workers = 4
    sets.model_depth = 10
    sets.resnet_shortcut = 'A'
    sets.input_D = 210  # Z
    sets.input_H = 140  # Y
    sets.input_W = 150  # X
    sets.learning_rate = 0.001

    # Check /change
    if sets.augmentation == 'True':
        sets.label_list = './toy_data/DRF_label_augmented_sets/set_{}/train.txt'.format(sets.setnr)
        sets.im_dir = './toy_data/DRF_augmented_sets/set_{}/'.format(sets.setnr)
        #sets.val_dir = './toy_data/DRF_augmented_sets/set_{}/validation'.format(sets.setnr)
        sets.method = 'method{}_a_v{}'.format(sets.methodnr, sets.version)
    else:
        sets.label_list = './toy_data/DRF_label_sets/set_{}/train_new.txt'.format(sets.setnr)
        sets.label_list_val = './toy_data/DRF_label_sets/set_{}/validation.txt'.format(sets.setnr)
        sets.im_dir = './toy_data/DRF_sets/Images/'.format(sets.setnr)
        #sets.val_dir = './toy_data/DRF_sets/set_{}/validation/'.format(sets.setnr)
        sets.method = 'method{}_v{}'.format(sets.methodnr, sets.version)

    # Check / change method 2
    if sets.methodnr == '2':
        sets.pretrained = False
    else:
        sets.pretrain_path = './pretrain/resnet_10.pth'
        sets.pretrained = True

    # Set save and results files
    results_file = "results/{}_{}_{}_set{}_{}.log".format(sets.model, sets.model_depth, sets.phase, sets.setnr,
                                                       sets.method)
    sets.save_folder = "./trails/DRF_models/{}_{}_set{}_{}".format(sets.model, sets.model_depth, sets.setnr,
                                                                sets.method)

    # Set logging files
    logger = logging.getLogger('mylogger')
    handler = logging.FileHandler(results_file, mode='w')
    logger.addHandler(handler)

    # getting mode
    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets)
    params = [{'params': parameters, 'lr': sets.learning_rate}]

    # optimizer
    if sets.pretrained:
        params = [
            {'params': parameters['base_parameters'], 'lr': sets.learning_rate},
            {'params': parameters['new_parameters'], 'lr': 0}]

        # {'params': parameters['new_parameters'], 'lr': sets.learning_rate * 100}]
    else:
        params = [{'params': parameters, 'lr': sets.learning_rate}]

    # optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)
    optimizer = torch.optim.Adam(params, sets.learning_rate)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma = 1)
    scheduler = None
    logger.info(
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

    training_dataset = DRF_data(sets.im_dir, sets.label_list, sets)
    validation_dataset = DRF_data(sets.im_dir, sets.label_list_val, sets)
    
    data_loader = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=False, num_workers=sets.num_workers,
                             pin_memory=sets.pin_memory)
    
    validation_loader = DataLoader(validation_dataset, batch_size=sets.batch_size, shuffle=False, num_workers=sets.num_workers,
                             pin_memory=sets.pin_memory)
    # training
    train(data_loader, validation_loader, model, optimizer, scheduler, total_epochs=sets.n_epochs, save_interval=sets.save_intervals,
          save_folder=sets.save_folder, sets=sets)
    