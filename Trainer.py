#-*-coding:utf-8-*-
'''
@FileName:
    Trainers.py
@Description:
    train and evaluate network
@Authors:
    Hanbo Sun(sun-hb17@mails.tsinghua.edu.cn)
@CreateTime:
    2020/04/24 02:16
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import DataParallel
import torch.utils.data as Data
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator

import Dataset
import Network

class DataLoaderX(Data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Trainer(object):
    '''
    trainer
    '''
    def __init__(self):
        # datasets
        self.train_dataset = Dataset.UCF101VideoDataset(
            '/home/sunhanbo/workspace/C3D_UCF101/Dataset', 'train', 16, False
        )
        self.train_loader = DataLoaderX(
            self.train_dataset, batch_size = 60, shuffle = True, drop_last = True, num_workers = 16,
        )
        self.test_dataset = Dataset.UCF101VideoDataset(
            '/home/sunhanbo/workspace/C3D_UCF101/Dataset', 'test', 16, False
        )
        self.test_loader = DataLoaderX(
            self.test_dataset, batch_size = 60, shuffle = False, drop_last = False, num_workers = 16,
        )
        # networks
        self.net = Network.C3DNetwork(101)
        origin_net_weights = torch.load(
            './zoo/dct_c3d_ucf101_120_0.9529780564263323.pth',
            map_location = lambda storage, location: storage
        )
        origin_net_weights['frequency_weights'] = torch.zeros(16)
        self.net.load_state_dict(origin_net_weights, strict = True)
        self.device = [4, 5, 6, 7]
        self.frequency_lambda = 1e-2
    def train(self):
        '''
        train network
        '''
        # set net on gpu
        self._load_net_device()
        # loss and optimizer, param
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 5e-4)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = [60, 90], gamma = 0.1)
        # init test
        init_accuracy = self.evaluate()
        print(f'init accuracy is {init_accuracy}')
        # epochs
        for epoch in range(120):
            # train
            self.net.train()
            tqdm_loader = tqdm(self.train_loader)
            for images, labels in tqdm_loader:
                outputs = self.net(images)
                labels = labels.to(outputs.device)
                # loss and backward
                loss = criterion(outputs, labels)
                # add frequency weights
                frequency_loss = torch.sum(torch.sigmoid(self.net.module.frequency_weights))
                loss = loss + self.frequency_lambda * frequency_loss
                self.net.zero_grad()
                loss.backward()
                optimizer.step()
                tqdm_loader.set_description(f'celoss is {loss.item():.4f}, floss is {frequency_loss.item():.4f}')
            scheduler.step()
            # test and scheduler
            if epoch == 0 or (epoch + 1) % 10 == 0:
                accuracy = self.evaluate()
                print(f'epoch {epoch+1:04d}: accuracy is {accuracy}')
                torch.save(self.net.module.state_dict(), f'./zoo/reweight_dct_c3d_ucf101_{epoch+1}_{accuracy}.pth')
    def evaluate(self):
        '''
        evaluate network
        '''
        self._load_net_device()
        self.net.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader):
                outputs = self.net(images)
                _, predicted = torch.max(outputs, 1)
                # test
                labels = labels.to(outputs.device)
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)
            reweights = self.net.module.frequency_weights
            reweights = torch.sigmoid(reweights).detach().cpu().numpy()
        print(f'In this epoch, frequency weights are {reweights}')
        return test_correct / test_total
    def _load_net_device(self):
        '''
        load net to device
        '''
        if not isinstance(self.net, DataParallel):
            self.net.to(torch.device(f'cuda:{self.device[0]}'))
            self.net = DataParallel(self.net, device_ids = self.device)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
