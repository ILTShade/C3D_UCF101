import math
import os
import shutil

import cv2
import numpy as np
import torch
import torch.utils.data as Data


class UCF101VideoDataset(Data.Dataset):
    def __init__(self, root_dir, split, clip_len = 16, preprocess = False):
        self.dataset_dir = os.path.join(root_dir, 'UCF101')
        self.input_dir = os.path.join(root_dir, 'UCF101_OUTPUT')
        # param
        self.train_ratio = 0.9
        self.resize_width = 171
        self.resize_height = 128
        self.frame_minimum = 16
        self.max_frequency = 4
        self.clip_len = clip_len
        if preprocess or not self.check_preprocess():
            self.preprocess()
        # get fnames
        assert split in ['train', 'test']
        self.split = split
        self.fnames = list(map(
            lambda x: os.path.join(self.input_dir, split, x),
            os.listdir(os.path.join(self.input_dir, split))
        ))
        if len(self.fnames) == 0:
            raise Exception(f'empty list')
    def __len__(self):
        return len(self.fnames)
    def __getitem__(self, index):
        label = int(os.path.splitext(os.path.basename(self.fnames[index]))[0].split('_', 1)[0])
        frame = np.load(self.fnames[index]) # CTHW
        assert self.resize_height == frame.shape[2] and self.resize_width == frame.shape[3]
        # crop
        t_start = np.random.randint(frame.shape[1] - self.frame_minimum)
        h_start = np.random.randint(self.resize_height - self.clip_len)
        w_start = np.random.randint(self.resize_width - self.clip_len)
        frame = frame[
            :,
            t_start:(t_start+self.frame_minimum),
            h_start:(h_start+self.clip_len),
            w_start:(w_start+self.clip_len)
        ]
        # normalize
        frame = frame.astype(np.float)
        mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1, 1))
        std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1, 1))
        frame = (frame - mean) / std
        # random flip
        if self.split == 'train' and np.random.random() > 0.5:
            frame = frame[:,:,:,::-1]
        return frame, label
    def check_preprocess(self):
        '''
        check if there is already input dir
        '''
        return os.path.exists(self.input_dir)
    def preprocess(self):
        '''
        change avi to ndarray with label
        '''
        # mkdir
        if not os.path.exists(self.input_dir):
            os.mkdir(self.input_dir)
        if not os.path.exists(os.path.join(self.input_dir, 'train')):
            os.mkdir(os.path.join(self.input_dir, 'train'))
        if not os.path.exists(os.path.join(self.input_dir, 'test')):
            os.mkdir(os.path.join(self.input_dir, 'test'))
        # change avi to ndarray with label
        for category_count, category_dir in enumerate(os.listdir(self.dataset_dir)):
            print(f'preprocess category {category_count}')
            category_dir = os.path.join(self.dataset_dir, category_dir)
            if not os.path.isdir(category_dir):
                raise Exception(f'{category_dir} is not a dir')
            # list dir
            avi_list = list(map(lambda x: os.path.join(category_dir, x), os.listdir(category_dir)))
            split_index = 0 - int((1 - self.train_ratio) * len(avi_list))
            train_avi_list = avi_list[:split_index]
            test_avi_list = avi_list[split_index:]
            self.preprocess_video(train_avi_list, os.path.join(self.input_dir, 'train'), category_count)
            self.preprocess_video(test_avi_list, os.path.join(self.input_dir, 'test'), category_count)
            # process train or test video
    def preprocess_video(self, avi_list, output_dir, label):
        '''
        change avi to ndarray
        '''
        if len(avi_list) == 0:
            raise Exception('empty avi list')
        for avi in avi_list:
            print(f'preprocess {avi}')
            output_name = f'{label:04d}_' + os.path.splitext(os.path.basename(avi))[0] + '.npy'
            output_name = os.path.join(output_dir, output_name)
            if os.path.exists(output_name):
                continue
            capture = cv2.VideoCapture(avi)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # frequency
            extract_frequency = min(math.floor(frame_count / self.frame_minimum), self.max_frequency)
            resize_flag = not (frame_width == self.resize_width and frame_height == self.resize_height)
            # read
            frame_index = 0
            output_array = []
            while True:
                ret, frame = capture.read()
                if not ret:
                    if frame_index / frame_count >= 0.9:
                        break
                    raise Exception(f'can not read {avi}, {frame_index}/{frame_count}')
                if frame_index % extract_frequency == 0:
                    if resize_flag:
                        frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                    output_array.append(np.transpose(frame, (2, 0, 1)))
                frame_index = frame_index + 1
            capture.release()
            # stack and save with label
            output_array = np.stack(output_array, axis = 1)
            np.save(output_name, output_array)

if __name__ == '__main__':
    ucf101_dataset = UCF101VideoDataset('/home/sunhanbo/workspace/C3D_UCF101/Dataset', 'train', 16, True)
