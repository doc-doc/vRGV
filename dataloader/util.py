# ====================================================
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : util.py
# ====================================================
import json
import os
import os.path as osp
import numpy as np
import pickle as pkl

def load_file(file_name):

    with open(file_name, 'r') as fp:
        if osp.splitext(file_name)[1]== '.txt':
            annos = fp.readlines()
            annos = [line.rstrip() for line in annos]
        if osp.splitext(file_name)[1] == '.json':
            annos = json.load(fp)

    return annos

def pkload(file):
    with open(file, 'rb') as fp:
        data = pkl.load(fp)
    return data

def pkdump(data, file):
    dirname = osp.dirname(file)
    if not osp.exists(dirname):
        os.makedirs(dirname)
    with open(file, 'wb') as fp:
        pkl.dump(data, fp)

def get_video_frames(video_relation_file):

    folders = load_file(video_relation_file)
    vframes = {}
    for recode in folders:
        video, nframe = recode[0], recode[1]
        if video not in vframes:
            vframes[video] = nframe
        else:
            continue

    all_frames = []
    sample_num = 120

    for video, nframe in vframes.items():
        
        samples = np.round(np.linspace(
            0, nframe-1, sample_num))

        samples = set([int(s) for s in samples])
        samples = list(samples)
        fnames = [osp.join(video, str(fid).zfill(6)) for fid in samples]
        if all_frames == []:
            all_frames = fnames
        else:
            all_frames.extend(fnames)

    return all_frames

def select_bbox(roi_bbox, roi_classme, width, height):
        """
        select the bboxes with maximun confidence
        :param roi_bbox:
        :param roi_classme:
        :return:
        """
        bbox, classme = roi_bbox.squeeze(), roi_classme.squeeze()
        classme = classme[:, 1:]  # skip background
        index = np.argmax(classme, 1)
        bbox = np.asarray([bbox[i][4 * (index[i] + 1):4 * (index[i] + 1) + 4] for i in range(len(bbox))])
        relative_bbox = bbox / np.asarray([width, height, width, height])
        area = (bbox[:,2]-bbox[:,0]+1)*(bbox[:,3]-bbox[:,1]+1)
        relative_area = area/(width*height)
        relative_area = relative_area.reshape(-1, 1)
        relative_bbox = np.hstack((relative_bbox, relative_area))

        return relative_bbox


def get_video_feature(video_feature_path, cache_file, frame_count, 
                    width, height, nbbox, frame_steps, feat_dim):
        """
        :param video_name:
        :param frame_count:
        :param width:
        :param height:
        :return:
        """
        # video_feature_folder = osp.join(video_feature_path, video_name)
        # cache_file = osp.join(video_feature_cache, '{}.npy'.format(video_name))
        # if osp.exists(cache_file) and osp.getsize(cache_file) > 0:
        #     video_feature = pkload(cache_file)
        #     return video_feature
        sample_frames = np.round(np.linspace(0, frame_count - 1, frame_steps))
        video_feature = np.zeros((len(sample_frames), nbbox, feat_dim), dtype=np.float32)
        for i, fid in enumerate(sample_frames):
            frame_name = osp.join(video_feature_path, str(int(fid)).zfill(6)+'.pkl')
            with open(frame_name, 'rb') as fp:
                feat = pkl.load(fp)
            roi_feat = feat['roi_feat'] #40x2048
            roi_bbox = feat['bbox']
            roi_classme = feat['cls_prob'] #40 x 81
            bbox = select_bbox(roi_bbox, roi_classme, width, height) # 40 x 5
            cb_feat = np.hstack((roi_feat, bbox))

            video_feature[i] = cb_feat

        np.savez(cache_file, x=video_feature)

        return video_feature
