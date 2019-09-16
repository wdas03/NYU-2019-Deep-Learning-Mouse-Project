import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import adabound
from torch.utils.data import Dataset
from sync_batchnorm import convert_model
import datetime
import os

def save_1(folder, model_1, optimizer, logger, epoch, scheduler):
    if not os.path.exists('../' + folder):
        os.makedirs('../' + folder)
        
    model_save_path = '../' + folder + '/' + str(datetime.datetime.now())+ ' ' + 'epoch: ' + str(epoch) + '.pth'

    state = {'epoch': epoch, 'state_dict_1': model_1.state_dict(),
        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'logger': logger}

    torch.save(state, model_save_path)
    
    print('Checkpoint {} saved !'.format(epoch))

    pass

def load_from_file_model_optimizer_scheduler(filename, model, optimizer, scheduler, data_parallel=True, sync_batch=False,):
    checkpoint = torch.load(filename)
    if data_parallel:
        model = nn.DataParallel(model)
    if sync_batch:
        model = convert_model(model)
    
    model.load_state_dict(checkpoint['state_dict_1'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    
    return model, optimizer, scheduler

