# ====================================================
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : data_preparation.py
# ====================================================

from util import *
import os.path as osp
import json
import pickle as pkl
import torch
import numpy as np
import os
import sys
sys.path.insert(0, '/storage/jbxiao/workspace/ground_code/lib')
from model.nms.nms_wrapper import nms



def load_predict(predict_file, topn):
    """
    nms within class ans then select the top n bbox of higher score among classes
    :param predict_file:
    :return:
    """
    with open(predict_file, 'rb') as fp:
        predict = pkl.load(fp)


    pred_boxes = predict['bbox'].squeeze()
    scores = predict['cls_prob'].squeeze()
    roi_feat = predict['roi_feat'].squeeze()

    pthresh = 0.00001
    bbox = []
    keep_inds = []
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
            keep = keep.cpu().data.numpy()
            if first:
                bbox = tmp_bbox
                keep_inds = keep
            else:
                bbox = np.vstack((bbox, tmp_bbox))
                keep_inds = np.vstack((keep_inds, keep))
            first = False


    rank_ind = bbox[:,-1].argsort()
    select_inds = keep_inds[rank_ind][-topn:]
    select_classme = scores[select_inds, :].squeeze()
    select_feat = roi_feat[select_inds, :].squeeze()
    select_bbox = bbox[rank_ind][-topn:, 0:4]
    return select_bbox, select_classme.cpu().data.numpy(), select_feat.cpu().data.numpy()


def select_feature(predict_dir, video_list, save_dir):
    """
    select bbox from the 1000 region proposals
    :param predict_dir:
    :param video_list:
    :return:
    """
    videos = load_file(video_list)
    for vid, vname in enumerate(videos):
        if vid <= 600: continue
        if vid > 800: break

        body_name = osp.splitext(vname)[0]
        predict_file = osp.join(predict_dir, body_name)
        files = os.listdir(predict_file)
        save_folder = osp.join(save_dir, body_name)

        if not osp.exists(save_folder):
            os.makedirs(save_folder)
        for file in files:
            path = osp.join(predict_file, file)
            bbox, classme, feat = load_predict(path, 40)
            # print(bbox.shape, classme.shape, feat.shape)
            feature  = {'bbox': bbox, 'classme': classme, 'feat': feat}
            save_file = osp.join(save_folder, file)
            with open(save_file, 'wb') as fp:
                pkl.dump(feature, fp)

        print(vid, save_folder)


def get_video_relation(anno_dir, video_list, mode):
    """
    obtain video relation samples
    :param anno_dir:
    :param video_list:
    :return:
    """
    videos = load_file(video_list)
    vrelations = []
    for video in videos:
        basename = osp.splitext(video)[0]
        path = osp.join(anno_dir, video)
        anno = load_file(path)
        id2cls = {}
        subobj = anno['subject/objects']
        for item in subobj:
            id2cls[item['tid']] = item['category']

        frame_count = anno['frame_count']
        frame_width, frame_height = anno['width'], anno['height']
        relations = anno['relation_instances']
        for rel in relations:
            subject = id2cls[rel['subject_tid']]
            object = id2cls[rel['object_tid']]
            predicate = rel['predicate']
            relation = '-'.join([subject, predicate, object])
            vrelations.append((basename, frame_count, frame_width,  frame_height,relation))

    save_file = '../dataset/vidvrd/vrelation_{}.json'.format(mode)
    print('save to {}'.format(save_file))
    with open(save_file, 'w') as fp:
        json.dump(vrelations, fp)



def main():
    root_dir = '/storage/jbxiao/workspace/'
    predict_dir = root_dir + 'ground_data/new_dets/'
    ground_dir = root_dir + 'vdata/'
    anno_dir = osp.join(ground_dir, 'vidvrd')
    train_list = osp.join(ground_dir, 'train_list.txt')
    val_list = osp.join(ground_dir, 'val_list.txt')
    save_dir = osp.join(root_dir, 'ground_data/video_feature')

    select_feature(predict_dir, train_list, save_dir)

    # get_video_relation(anno_dir, train_list, 'train')


if __name__ == "__main__":
    main()
