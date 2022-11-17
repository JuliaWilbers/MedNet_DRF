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
import pandas as pd

def accuracy_fn(y_true, y_pred, batch_size):
    correct = torch.eq(y_true, y_pred).sum().item() 
    acc = (correct / batch_size) * 100 
    return acc

def test(results_test_file, data_loader, model, sets):
    model.eval() # for testing 
    loss_clas = nn.BCELoss()
    
    pl = []
    ypl =[]
    tl =[]
    
    for batch_id, batch_data in enumerate(data_loader):
        # forward
        volumes, masks, labels, name = batch_data
        if not sets.no_cuda:
            volume = volume.cuda()
            labels = labels.cuda()
            masks = masks.cuda()
        with torch.no_grad():
            y_pb = model(volumes, masks)
            y_pred = torch.round(y_pb)
            y_pred = y_pred.squeeze()
            y_true = labels.squeeze()
            
            test_loss = loss_clas(y_pb, y_true)
            test_acc = accuracy_fn(y_true, y_pred, sets.batch_size)
            
            print(f"Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
            
            pl.append(name)
            tl.append(labels.item())
            ypl.append(y_pb.item())
            
    #store results
    df = pd.DataFrame(list(zip(*[pl, tl, ypl])), columns = ['Patient', 'true label', 'posterior prob'])
    df.to_csv(results_test_file, index=False)        
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
    
    # Change
    sets.data_dir = './data/DRF_data/'
    sets.label_list = sets.label_list = './data/DRF_label_sets/set_{}/test.txt'.format(sets.setnr)
    sets.set_name = 'set_{}'.format(sets.setnr)
    
    if sets.augmentation == True:
        sets.method ='method{}_a_v{}'.format(sets.methodnr, sets.version)
    else:
        sets.method ='method{}_v{}'.format(sets.methodnr, sets.version)
    
    results_test_file = "./results/{}_{}_{}.csv".format(sets.method, sets.phase, sets.set_name)
    
    # getting model
    print ('loading trained model {}'.format(sets.resume_path))
    checkpoint = torch.load(sets.resume_path)
    net, _ = generate_model(sets)
    net.load_state_dict(checkpoint['state_dict'], strict=False)

    # data tensor
    testing_data =DRF_data(sets.data_dir, sets.label_list, sets)
  
    data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)
    
    acc, test_loss = test(results_test_file, data_loader, net, sets)

   

