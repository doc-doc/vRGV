# ====================================================
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : detect_frame.py
# ====================================================

import os
import sys
sys.path.insert(0, 'lib')
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
from collections import defaultdict
import os.path as osp
import pickle as pkl

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.resnet import resnet
import pdb
import threading




class FeatureExtractor():
    def __init__(self, train_loader, val_loader, cfg_file, classes,
                 class_agnostic, cuda, checkpoint_path):
        self.cfg_file = cfg_file
        self.classes = classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_agnostic = class_agnostic
        self.cuda = cuda
        self.load_name = checkpoint_path
        self.max_per_image = 100
        self.pthresh = 0



    def build_model(self):
        self.fasterRCNN = resnet(self.classes, 101, pretrained=False, class_agnostic=self.class_agnostic)
        self.fasterRCNN.create_architecture()


    def load_checkpoint(self):
        print('Load checkpoint from {}'.format(self.load_name))
        if self.cuda > 0:
            print('use gpu True')
            checkpoint = torch.load(self.load_name)
        else:
            checkpoint = torch.load(self.load_name, map_location=(lambda storage, loc: storage))
        self.fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']


    def run(self):
        cfg_from_file(self.cfg_file)
        cfg.USE_GPU_NMS = self.cuda
        # print('Using config:')
        # pprint.pprint(cfg)
        np.random.seed(cfg.RNG_SEED)
        self.build_model()
        self.load_checkpoint()
        self.detect()



    def detect(self):
        im_data = torch.FloatTensor(1)
        im_info = torch.FloatTensor(1)
        num_boxes = torch.LongTensor(1)
        gt_boxes = torch.FloatTensor(1)

        # ship to cuda
        if self.cuda > 0:
            im_data = im_data.cuda()
            im_info = im_info.cuda()
            num_boxes = num_boxes.cuda()
            gt_boxes = gt_boxes.cuda()

            cfg.CUDA = True
            self.fasterRCNN = self.fasterRCNN.cuda()

        # make variable
        with torch.no_grad():
            im_data = Variable(im_data)
            im_info = Variable(im_info)
            num_boxes = Variable(num_boxes)
            gt_boxes = Variable(gt_boxes)

        self.fasterRCNN.eval()

        save_dir = '../ground_data/frame_feature/'

        for iv, inputs in enumerate(self.val_loader):
            #if iv <= 200000: continue
            # if iv > 200000: break

            spatial_data, frame_name = inputs
            frame_name = frame_name[0]

            save_name = osp.join(save_dir, frame_name + '.pkl')
            if osp.exists(save_name): continue

            fdet = self.get_snippet_dets(spatial_data, im_data, im_info, gt_boxes, num_boxes)


            save_name = save_dir + frame_name + '.pkl'
            dirname = osp.dirname(save_name)
            if not osp.exists(dirname):
                os.makedirs(dirname)

            with open(save_name, 'wb') as fp:
                pkl.dump(fdet, fp)
            if iv % 500 == 0:
                print(iv, save_name)



    def get_snippet_dets(self, spatial_data, im_data, im_info, gt_boxes, num_boxes):
        """
        get detection results for each frame in the snippet
        :param spatial_data:
        :param im_data:
        :param im_info:
        :param gt_boxes:
        :param num_boxes:
        :return:
        """
        fdet = {}
        im_blob, im_scale = spatial_data['im_blob'][0], spatial_data['im_scale'][0]
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scale[0]]], dtype=np.float32)

        im_data_pt = im_blob
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.data.resize_(1, 1, 5).zero_()
        num_boxes.data.resize_(1).zero_()

        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, pooled_feat, base_feat = self.fasterRCNN(im_data, im_info, gt_boxes, num_boxes)


        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        pred_boxes = self.transform_bbox(boxes, bbox_pred, im_info, scores)
        assert im_scale[0].item() != 0, "im_scale==0"

        pred_boxes /= im_scale[0].item()

        fdet['cls_prob'] = cls_prob.cpu().data.numpy()
        fdet['bbox'] = pred_boxes.cpu().data.numpy()
        fdet['roi_feat'] = pooled_feat.cpu().data.numpy()
        # fdet['base_feat'] = base_feat.cpu().data.numpy()

        return fdet


    def transform_bbox(self, boxes, bbox_pred, im_info, scores):
        """
        transform bbox from
        :param boxes:
        :param bbox_pred:
        :param im_info:
        :param scores:
        :return:
        """
        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if self.class_agnostic:
                    if self.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    if self.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4 * 81)

            # print(boxes.shape, box_deltas.shape)

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        return pred_boxes


    def bbox_selection(self, relation, pred_boxes, scores, pooled_feat):
        """
        delete bbox of low scores and do NMS
        :param pred_boxes:
        :param scores:
        :return:
        """

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        class_bboxes = {}
        class_feats = {}
        class_pros = {}
        sub, pre, obj = relation[0].split('-')
        sind, oind = self.classes.index(sub), self.classes.index(obj)
        for c, j in enumerate([sind, oind]):
            inds = torch.nonzero(scores[:, j] > self.pthresh).view(-1)

            if inds.numel() > 0:

                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if self.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)

                cls_dets = cls_dets[order]
                keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)

                inds =  keep.view(-1).long()
                #inds = inds.numpy()
                inds_new = []
                for i in inds:
                    bbox = cls_dets[i, 0:4]
                    if bbox[2]-bbox[1] < 5.0 or bbox[3]-bbox[1] < 5.0:
                        continue
                    inds_new.append(i)

                if len(inds_new) == 0: continue

                inds = torch.cuda.LongTensor(np.array(inds_new))

                cls_dets = cls_dets[inds]
                cls_feats = pooled_feat[inds]
                cls_pros = scores[inds]

                class_bboxes[c] = cls_dets.data.cpu().numpy()
                class_feats[c] = cls_feats.data.cpu().numpy()
                class_pros[c] = cls_pros.data.cpu().numpy()
                if c == 0 and sind == oind:
                    class_bboxes[1] = cls_dets.data.cpu().numpy()
                    class_feats[1] = cls_feats.data.cpu().numpy()
                    class_pros[1] = cls_pros.data.cpu().numpy()
                    break

                """
                print('bbox shape:{}\t classme shape:{}\tfeat shape:{}'.format(cls_dets.shape,
                                                                               cls_feats.shape,
                                                                               cls_pros.shape))
                """
        return class_bboxes, class_pros, class_feats
