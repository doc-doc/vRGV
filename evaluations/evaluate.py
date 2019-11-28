# ====================================================
# @Time    : 8/9/19 4:29 PM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : evaluate.py
# ====================================================
from util import *
import os
import os.path as osp
import h5py
import json
import pickle as pkl
import torch
import numpy as np
import os
import sys
sys.path.insert(0, '../lib')
from model.nms.nms_wrapper import nms


def array2matrix(position):
    row = position/80
    col = position%80
    return row, col


def load_predict(predict_file):
    with open(predict_file, 'rb') as fp:
        predict = pkl.load(fp)
    pred_boxes = predict['bbox'].squeeze()
    scores = predict['cls_prob'].squeeze()

    pthresh = 0.00001
    bbox = []
    first = True
    for j in range(81):
        if j == 0: continue # skip the background
        inds = torch.nonzero(scores[:, j] > pthresh).view(-1)
        if len(inds) == 0: continue

        cls_scores = scores[:, j][inds]
        _, order = torch.sort(cls_scores, 0, True)

        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)

        cls_dets = cls_dets[order]
        keep = nms(cls_dets, 0.4, force_cpu=0)

        inds = keep.view(-1).long()
        if len(inds) > 0:
            tmp_bbox = cls_dets[inds].cpu().data.numpy()
            if first:
                bbox = tmp_bbox
            else:
                bbox = np.vstack((bbox, tmp_bbox))
            first = False

        # if len(inds) > 0:
        #     tmp_bbox = cls_dets[inds]
        #
        #     if first:
        #         bbox = tmp_bbox
        #     else:
        #         bbox = torch.cat((bbox, tmp_bbox))

        #     first = False

    topn = 100
    select_bbox = bbox[bbox[:,-1].argsort()][-topn:, 0:4]
    # select_bbox = bbox[torch.sort(bbox[:, -1])[1]][-topn:, 0:4]
    # select_bbox = select_bbox.data.numpy()
    print(select_bbox.shape)
    return select_bbox



def frame_recall(predict_bbox, fdet):

    recall = 0
    for anno_b in fdet:
        bbox = anno_b['bbox']
        bbox = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
        # print(bbox)
        for pred_bbox in predict_bbox:
            # print(pred_bbox.shape, len(predict_bbox))
            op = iou(bbox, pred_bbox)
            if op >= 0.5:
                recall += 1
                break
    return recall


def recall(predict_dir, anno_dir, video_list):
    """

    :param predict_dir:
    :param anno_dir:
    :param train_list:
    :return:
    """
    videos = load_file(video_list)
    total_bbox = 0
    total_recall = 0

    for vid, vname in enumerate(videos):
        body_name = osp.splitext(vname)[0]
        path = osp.join(anno_dir, vname)
        anno = load_file(path)
        tracklet = anno['trajectories']
        vrecall = 0
        vbbox = 0
        for fid, fdet in enumerate(tracklet):
            if fdet == []: continue
            frame_predict_file = osp.join(predict_dir+body_name, '{:06d}.pkl'.format(fid))
            predict_bbox = load_predict(frame_predict_file)
            ground_f = len(fdet)
            recall = frame_recall(predict_bbox, fdet)
            vbbox += ground_f
            vrecall += recall
        total_bbox += vbbox
        total_recall += vrecall

        print('{}:{}:{}'.format(vid, vname, vrecall/vbbox))

    print(total_recall/total_bbox)


def main():
    root_dir = '/storage/jbxiao/workspace/'
    predict_dir = root_dir + 'ground_data/new_dets/'
    ground_dir = root_dir + 'vdata/'
    anno_dir = osp.join(ground_dir, 'vidvrd')
    train_list = osp.join(ground_dir, 'train_list.txt')
    val_list = osp.join(ground_dir, 'val_list.txt')

    recall(predict_dir, anno_dir, val_list)


if __name__ == "__main__":
    main()
