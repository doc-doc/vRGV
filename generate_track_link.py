# ====================================================
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : generate_track.py
# ====================================================
from tools.util import load_file
import os.path as osp
import numpy as np
import pickle as pkl
from utils import save_results, sort_bbox, pkload, pkdump
from tube import *
import time

sample_fnum = 120
beta_thresh = 0.04

def load_video_bbox(vname, feat_dir, nframe):
    """
    load bboxes for a video
    :param vname:
    :param feat_dir:
    :param nframe:
    :return:
    """

    video_feature_folder = osp.join(feat_dir, vname)
    sample_frames = np.round(np.linspace(0, nframe - 1, sample_fnum))
    sample_frames = [int(num) for num in sample_frames]
    videos = []
    for i, fid in enumerate(sample_frames):
        frame_name = osp.join(video_feature_folder, str(fid).zfill(6) + '.pkl')
        with open(frame_name, 'rb') as fp:
            feat = pkl.load(fp)
        bbox, classme = feat['bbox'].squeeze(), feat['cls_prob'].squeeze()
        classme = classme[:, 1:] #skip background
        index = np.argmax(classme, 1)
        bbox = np.asarray([bbox[i][4*(index[i]+1):4*(index[i]+1) + 4] for i in range(len(bbox))])
        videos.append(bbox)
    return videos, sample_frames


def interpolate(sub_bboxes, obj_bboxes, valid_frame_idx, sample_frames, nframe):
    """
    linear interpolate the missing bboxes
    :param sub_bboxes:
    :param obj_bboxes:
    :param valid_frames:
    :param nframe:
    :return:
    """
    sub_bboxes = np.asarray(sub_bboxes)
    obj_bboxes = np.asarray(obj_bboxes)

    full_sub_bboxes = []
    full_obj_bboxes = []

    for i, id in enumerate(valid_frame_idx):

        full_sub_bboxes.append(sub_bboxes[i])
        full_obj_bboxes.append(obj_bboxes[i])
        if i == len(valid_frame_idx)-1: break

        pre_frame = sample_frames[id]
        next_frame = sample_frames[id+1]
        gap = next_frame - pre_frame
        if gap == 1: continue
        for mid in range(pre_frame+1, next_frame):
            sub_bbox = (next_frame - mid) / gap * sub_bboxes[i] + (mid - pre_frame) / gap * sub_bboxes[i + 1]
            obj_bbox = (next_frame - mid) / gap * obj_bboxes[i] + (mid - pre_frame) / gap * obj_bboxes[i + 1]
            full_sub_bboxes.append(sub_bbox)
            full_obj_bboxes.append(obj_bbox)

    fnum = sample_frames[valid_frame_idx[-1]]-sample_frames[valid_frame_idx[0]]+1
    assert len(full_sub_bboxes) == fnum, 'interpolate error'
    full_sub_bboxes = [bbox.tolist() for bbox in full_sub_bboxes]
    full_obj_bboxes = [bbox.tolist() for bbox in full_obj_bboxes]

    return full_sub_bboxes, full_obj_bboxes


def generate_track(val_list_file, results_file, feat_dir, bbox_dir, res_file):
    """
    generate tracklet from attention value
    :param val_list_file:
    :param results_dir:
    :return:
    """
    val_list = load_file(val_list_file)
    total_n = len(val_list)
    pre_vname = ''
    results, video_bboxes = None, None
    sample_frames = None
    results_all = load_file(results_file)

    final_res = {}
    video_res = {}

    for i, sample in enumerate(val_list):

        vname, nframe, width, height, relation = sample

        # if vname != 'ILSVRC2015_train_00267002': continue
        # if relation.split('-')[0] == relation.split('-')[-1]: continue
        # if nframe <= 120: continue
        if vname != pre_vname:
            cache_file = osp.join(bbox_dir, vname + '.pkl')
            data = pkload(cache_file)
            if not (data is None):
                video_bboxes, sample_frames = data
            else:
                video_bboxes, sample_frames = load_video_bbox(vname, feat_dir, nframe)
                pkdump((video_bboxes, sample_frames), cache_file)
            results = results_all[vname]
            if i > 0:
                final_res[pre_vname] = video_res
            video_res = {}
            print(i, vname)

        alpha_s = np.array(results[relation]['sub'])
        alpha_o = np.array(results[relation]['obj'])

        beta1 = results[relation]['beta1']
        beta2 = results[relation]['beta2']

        # print(alpha_o.shape, beta1.shape)

        nsample, nclip = len(beta1), len(beta2)
        beta1 = np.asarray(beta1)
        beta2 = np.asarray(beta2)
        step = nsample//nclip
        temp = np.zeros(nsample)
        for cp in range(nclip):
            temp[cp*step:(cp+1)*step] = beta2[cp] + beta1[cp*step:step*(cp+1)]

        t1 = time.time()
        sub_bboxes, obj_bboxes, sid, valid_frame_idx = link_bbox(video_bboxes, alpha_s, alpha_o,
                                                                 temp, beta_thresh,sample_frames, nframe)
        t2 = time.time()
        if valid_frame_idx is None:
            sub_bboxes = {}
            obj_bboxes = {}
        else:
            if nframe > sample_fnum:
                sub_bboxes, obj_bboxes = interpolate(sub_bboxes,obj_bboxes,valid_frame_idx,sample_frames,nframe)

            sid = sample_frames[sid]
            sub_bboxes = {fid+sid:bbox for fid, bbox in enumerate(sub_bboxes)}
            obj_bboxes = {fid+sid:bbox for fid, bbox in enumerate(obj_bboxes)}

        ins = {"sub": sub_bboxes, "obj": obj_bboxes}
        video_res[relation] = ins
        # vis_prediction_online(ins, vname, relation)
        pre_vname = vname

        if i == total_n -1:
            final_res[vname] = video_res

    save_results(res_file, final_res)


def main(res_file):
    data_dir = '../ground_data/'
    dataset = 'vidvrd'
    val_list_file = 'dataset/{}/vrelation_val.json'.format(dataset)
    result_file = '../{}/results/{}_batch.json'.format(data_dir, dataset)

    feat_dir = osp.join(data_dir, dataset, 'frame_feature')
    bbox_dir = osp.join(data_dir, dataset, 'bbox')
    generate_track(val_list_file, result_file, feat_dir, bbox_dir, res_file)


if __name__ == "__main__":
    res_file = 'results/test_viterbi_1gap_04_batch.json'
    main(res_file)
