# ====================================================
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : detect_frame_loader.py
# ====================================================
import sys
sys.path.insert(0, 'lib')
from .util import *
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os.path as osp
import numpy as np
import cv2
from model.utils.config import cfg, cfg_from_file, cfg_from_list



class ExpDataset(Dataset):
    """load the dataset in dataloader"""
    def __init__(self, dic, anno_path, spatial_path, mode):
        self.frames = dic
        self.anno_path = anno_path
        self.spatial_path = spatial_path
        self.mode = mode

        
    def __len__(self):
        return len(self.frames)


    def transform_image(self, im):
        """resize and save scale"""
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS

        self.im_shape = im_orig.shape
        im_size_min = np.min(self.im_shape[0:2])
        im_size_max = np.max(self.im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)

            if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
                im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        return np.array(processed_ims).squeeze(), np.array(im_scale_factors,dtype=np.float32)


    def load_rgb_image(self, frame_name):
        """loading image from path and frame number"""
        full_name = osp.join(self.spatial_path, frame_name+'.JPEG')

        if not osp.exists(full_name):
            print('File {} not find'.format(full_name))
            return None
        img = Image.open(full_name)
        im_in = np.array(img)
        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)
        # rgb -> bgr
        im = im_in[:, :, ::-1]

        blob, im_scale = self.transform_image(im)

        return blob, im_scale


    def __getitem__(self, idx):
        #if idx <= 200000: return -1, -1
        # if idx > 200000: return -1, -1

        frame_name = self.frames[idx]
        blob, scale = self.load_rgb_image(frame_name)
        spatial_data = {}
        spatial_data['im_blob'] = [blob]
        spatial_data['im_scale'] = [scale]

        return spatial_data, frame_name


class DetectFrameLoader():
    def __init__(self, batch_size, num_worker, spatial_path,
                 dataset, train_list_path, val_list_path,
                 train_shuffle=True, val_shuffle=False):

        self.batch_size = batch_size
        self.num_workers = num_worker
        self.spatial_path = spatial_path


        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle

        self.dataset = dataset

        self.get_frames(train_list_path, val_list_path)


    def get_frames(self, train_list_path, val_list_path):

        train_list = get_video_frames(train_list_path)
        test_list = get_video_frames(val_list_path)

        self.train_frames = train_list
        self.val_frames = test_list


    def run(self):

        train_loader = '' # self.train()
        val_loader = self.validate()
        return train_loader, val_loader


    def train(self):
        #print("Now in train")
        #applying trabsformation on training videos 
        training_set = ExpDataset(dic=self.train_frames, anno_path=self.dataset, spatial_path=self.spatial_path,
                                               mode='train')

        print('Eligible frames for training :', len(training_set), 'video frames')
        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers)
        return train_loader


    def validate(self):
        #print("Now in Validate")
        #applying transformation for validation videos 
        validation_set = ExpDataset(dic=self.val_frames, anno_path=self.dataset, spatial_path=self.spatial_path,
                                                 mode='val')
        
        print('Eligible frames for validation:', len(validation_set), 'video frames')
        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.batch_size,
            shuffle=self.val_shuffle,
            num_workers=self.num_workers)

        return val_loader


