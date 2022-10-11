from setting import parse_opts 
from datasets.DRF_data import DRF_data
from DRF_model import generate_model
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy import ndimage
import nibabel as nib
import sys
import os
from utils.file_process import load_lines
import numpy as np
from torch import nn
import logging

def accuracy_fn(y_true, y_pred, batch_size):
    correct = torch.eq(y_true, y_pred).sum().item() 
    acc = (correct / batch_size) * 100 
    return acc

def test(results_test_file, data_loader, model, sets):
    model.eval() # for testing 
    loss_clas = nn.BCELoss()
    
    logger = logging.getLogger('mylogger')
    handler = logging.FileHandler(results_test_file, mode = 'w')
    logger.addHandler(handler)
    
    for batch_id, batch_data in enumerate(data_loader):
        # forward
        volumes, labels, name = batch_data
        if not sets.no_cuda:
            volume = volume.cuda()
        with torch.no_grad():
            y_pb = model(volumes)
            y_pred = torch.round(y_pb)
            y_pred = y_pred.squeeze()
            y_true = labels.squeeze()
            
            test_loss = loss_clas(y_pred, y_true)
            test_acc = accuracy_fn(y_true, y_pred, sets.batch_size)
            
            print(f"Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
            
            
            logger.warning('Patient = {}, Post_prob = {}, Label = {}'.format(name, y_pb.item(), labels.item()))

            
    return test_acc, test_loss


if __name__ == '__main__':
    # settting
    sets = parse_opts()
    sets.target_type = "normal"
    sets.phase = 'test'
    sets.n_epochs = 1
    sets.no_cuda = True
    sets.num_workers = 4
    sets.model_depth = 10
    sets.resnet_shortcut = 'A'
    sets.input_D = 210 #Z
    sets.input_H = 140 #Y
    sets.input_W = 150 #X
    sets.batch_size= 1
    sets.resume_path = "./trails/DRF_models/resnet_10_set_1_method2_epoch_69_batch_0.pth.tar"
    sets.im_dir = "./toy_data/DRF_sets/set_1/test"
    sets.label_list = './toy_data/DRF_label_sets/set_1/test.txt'
    sets.set_name = 'set_1'
    sets.method ='method2'
    
    results_test_file = "results/{}_{}_{}_{}_{}.log".format(sets.model, sets.model_depth, sets.phase, sets.set_name, sets.method)
    
    # getting model
    print ('loading trained model {}'.format(sets.resume_path))
    checkpoint = torch.load(sets.resume_path)
    net, _ = generate_model(sets)
    net.load_state_dict(checkpoint['state_dict'], strict=False)

    # data tensor
    testing_data =DRF_data(sets.im_dir, sets.label_list, sets)
  
    data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)
    
    acc, test_loss = test(results_test_file, data_loader, net, sets)

   
