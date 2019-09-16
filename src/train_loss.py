import numpy as np
import torch
from torch.utils.data import Dataset
import datetime


def save(model_2, optimizer, logger, epoch):
    model_save_path = '../save/' + str(datetime.datetime.now()) + '.pth'

    state = {'epoch': epoch, 'state_dict_2': model_2.state_dict(), 
             'optimizer': optimizer.state_dict(), 'logger': logger}

    torch.save(state, model_save_path)

    print('Checkpoint {} saved !'.format(epoch))

    pass
