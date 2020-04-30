import os
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.dataset import YunpeiDataset
from utils.utils import sample_frames

def get_dataset(src1_data, src1_train_num_frames, src2_data, src2_train_num_frames, src3_data, src3_train_num_frames,
                tgt_data, tgt_test_num_frames, batch_size):
    print('Load Source Data')
    print('Source Data: ', src1_data)
    src1_train_data_fake = sample_frames(flag=0, num_frames=src1_train_num_frames, dataset_name=src1_data)
    src1_train_data_real = sample_frames(flag=1, num_frames=src1_train_num_frames, dataset_name=src1_data)
    print('Source Data: ', src2_data)
    src2_train_data_fake = sample_frames(flag=0, num_frames=src2_train_num_frames, dataset_name=src2_data)
    src2_train_data_real = sample_frames(flag=1, num_frames=src2_train_num_frames, dataset_name=src2_data)
    print('Source Data: ', src3_data)
    src3_train_data_fake = sample_frames(flag=0, num_frames=src3_train_num_frames, dataset_name=src3_data)
    src3_train_data_real = sample_frames(flag=1, num_frames=src3_train_num_frames, dataset_name=src3_data)

    print('Load Target Data')
    print('Target Data: ', tgt_data)
    tgt_test_data = sample_frames(flag=2, num_frames=tgt_test_num_frames, dataset_name=tgt_data)

    src1_train_dataloader_fake = DataLoader(YunpeiDataset(src1_train_data_fake, train=True),
                                            batch_size=batch_size, shuffle=True)
    src1_train_dataloader_real = DataLoader(YunpeiDataset(src1_train_data_real, train=True),
                                            batch_size=batch_size, shuffle=True)
    src2_train_dataloader_fake = DataLoader(YunpeiDataset(src2_train_data_fake, train=True),
                                            batch_size=batch_size, shuffle=True)
    src2_train_dataloader_real = DataLoader(YunpeiDataset(src2_train_data_real, train=True),
                                            batch_size=batch_size, shuffle=True)
    src3_train_dataloader_fake = DataLoader(YunpeiDataset(src3_train_data_fake, train=True),
                                            batch_size=batch_size, shuffle=True)
    src3_train_dataloader_real = DataLoader(YunpeiDataset(src3_train_data_real, train=True),
                                            batch_size=batch_size, shuffle=True)
    tgt_dataloader = DataLoader(YunpeiDataset(tgt_test_data, train=False), batch_size=batch_size, shuffle=False)
    return src1_train_dataloader_fake, src1_train_dataloader_real, \
           src2_train_dataloader_fake, src2_train_dataloader_real, \
           src3_train_dataloader_fake, src3_train_dataloader_real, \
           tgt_dataloader









