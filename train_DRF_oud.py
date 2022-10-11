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

#cudnn.benchmark = True
#print(torch.cuda.device_count())

def train(data_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets):
    # settings
    batches_per_epoch = len(data_loader)
    print("batches per epoch is" + str(batches_per_epoch))
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    #loss_seg = nn.CrossEntropyLoss(ignore_index=-1)
    #loss_clas = nn.MSELoss()
    #loss_clas = nn.CrossEntropyLoss(ignore_index=-1)
    loss_clas = nn.BCELoss()
    
    print("Current setting is:")
    print(sets)
    print("\n\n")
    if not sets.no_cuda:
        #loss_seg = loss_seg.cuda()
        loss_clas = loss_clas.cuda()

    model.train()
    train_time_sp = time.time()
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))

        scheduler.step()
        log.info('lr = {}'.format(scheduler.get_lr()))

        for batch_id, batch_data in enumerate(data_loader):
            
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes, labels = batch_data
            
            if not sets.no_cuda:
                volumes = volumes.cuda()

            optimizer.zero_grad()
            output = torch.round(model(volumes))
            
            # resize label
            """[n, _, d, h, w] = out_masks.shape
            new_label_masks = np.zeros([n, d, h, w])
            for label_id in range(n):
                label_mask = label_masks[label_id]
                [ori_c, ori_d, ori_h, ori_w] = label_mask.shape
                label_mask = np.reshape(label_mask, [ori_d, ori_h, ori_w])
                scale = [d * 1.0 / ori_d, h * 1.0 / ori_h, w * 1.0 / ori_w]
                label_mask = ndimage.interpolation.zoom(label_mask, scale, order=0)
                new_label_masks[label_id] = label_mask

            new_label_masks = torch.tensor(new_label_masks).to(torch.int64)
            if not sets.no_cuda:
                new_label_masks = new_label_masks.cuda()
            """
            # calculating loss
            # labels = torch.tensor([1, 0, 0, 1, 1])
            # labels = torch.cuda.FloatTensor([1], device='cuda')
            # labels = torch.FloatTensor([1])
            # output = torch.FloatTensor([output.data])
            # labels = torch.FloatTensor([labels.data])
          
            #labels.type(torch.FloatTensor)
            #out_masks.type(torch.FloatTensor)
          
           
            #loss_value_seg = loss_seg(out_masks, new_label_masks)
            loss_value_clas = loss_clas(output.squeeze(), labels.squeeze())

            #loss = loss_value_seg
            loss = loss_value_clas
            loss.backward()
            optimizer.step()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                'Batch: {}-{} ({}), loss = {:.3f}, loss_seg = {:.3f}, avg_batch_time = {:.3f}' \
                    .format(epoch, batch_id, batch_id_sp, loss.item(), loss_value_clas.item(), avg_batch_time))

            if not sets.ci_test:
                # save model
                if batch_id == 0 and batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                    # if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                    model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(save_folder, epoch, batch_id)
                    model_save_dir = os.path.dirname(model_save_path)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)

                    log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id))
                    torch.save({
                        'ecpoch': epoch,
                        'batch_id': batch_id,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        model_save_path)

    print('Finished training')
    if sets.ci_test:
        exit()


if __name__ == '__main__':
    # settting
    sets = parse_opts()
    if sets.ci_test:
        sets.img_list = './toy_data/test_DRF.txt'
        sets.im_dir = './toy_data/DRF_sets/set_1/train/'
        sets.seg_dir = './toy_data/Segmentations'
        sets.n_epochs = 1
        sets.no_cuda = True
        sets.data_root = '.toy_data'
        sets.pretrain_path = ''
        sets.num_workers = 4
        sets.model_depth = 10
        sets.resnet_shortcut = 'A'
        sets.input_D = 210 #Z
        sets.input_H = 140 #Y
        sets.input_W = 150 #X
        sets.batch_size= 6

    # getting model
    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets)
    #print(model)
    #print('parameters' +  str(parameters))
   
    # optimizer
    if sets.ci_test:
        params = [{'params': parameters, 'lr': sets.learning_rate}]
    else:
        params = [
            {'params': parameters['base_parameters'], 'lr': sets.learning_rate},
            {'params': parameters['new_parameters'], 'lr': sets.learning_rate * 100}
        ]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

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
    #training_dataset = DRF_data(sets.data_root, sets.im_dir, sets.seg_dir, sets.img_list, 61, sets)
    training_dataset = DRF_data(sets.im_dir, sets.seg_dir, sets.img_list, sets.label_list, 49, sets)
    data_loader = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=False, num_workers=sets.num_workers,
                             pin_memory=sets.pin_memory)
    
    # training
    train(data_loader, model, optimizer, scheduler, total_epochs=sets.n_epochs, save_interval=sets.save_intervals,
          save_folder=sets.save_folder, sets=sets)