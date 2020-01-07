# ====================================================
# @Time    : 12/21/19 10:33 PM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : generate_track.py
# ====================================================
from tools.util import load_file
import os.path as osp
import numpy as np
import pickle as pkl
from utils import save_results

sample_fnum = 120
beta_thresh = 1e-6

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
        # print(index)
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
    full_sub_bboxes = []
    full_obj_bboxes = []

    for i, id in enumerate(valid_frame_idx):

        full_sub_bboxes.append(sub_bboxes[id])
        full_obj_bboxes.append(obj_bboxes[id])
        if i == len(valid_frame_idx)-1: break

        pre_frame = sample_frames[id]
        next_frame = sample_frames[id+1]
        gap = next_frame - pre_frame
        if gap == 1: continue
        for mid in range(pre_frame+1, next_frame):
            sub_bbox = (next_frame - mid) / gap * sub_bboxes[id] + (mid - pre_frame) / gap * sub_bboxes[id + 1]
            obj_bbox = (next_frame - mid) / gap * obj_bboxes[id] + (mid - pre_frame) / gap * obj_bboxes[id + 1]
            full_sub_bboxes.append(sub_bbox)
            full_obj_bboxes.append(obj_bbox)

    assert len(full_sub_bboxes) == nframe, 'interpolate error'

    return full_sub_bboxes, full_obj_bboxes


def generate_track(val_list_file, results_dir, feat_dir):
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

    final_res = {}
    video_res = {}

    for i, sample in enumerate(val_list):
        vname, nframe, width, height, relation = sample

        if vname != pre_vname:
            video_bboxes, sample_frames = load_video_bbox(vname, feat_dir, nframe)
            results = load_file(osp.join(results_dir, vname + '.json'))
            if i > 0:
                final_res[pre_vname] = video_res
            video_res = {}

        subs = results[relation]['sub']
        objs = results[relation]['obj']
        beta = results[relation]['beta'][0] #(1, sample_fnum)
        print(vname, relation)
        print(beta)

        sub_index = np.argmax(subs, 1) #select bbox with maximum attention value
        obj_index = np.argmax(objs, 1)

        sub_bboxes, obj_bboxes = [], []
        valid_frame_idx = []

        #delete redundant and add temporal attention score as indicator
        for j, bboxes in enumerate(video_bboxes):
            # if beta[0][j] < beta_thresh: continue
            if j > 0 and sample_frames[j] == sample_frames[j-1]: continue
            sub_b = np.append(bboxes[sub_index[j]], beta[j])
            obj_b = np.append(bboxes[obj_index[j]], beta[j])
            sub_bboxes.append(sub_b)
            obj_bboxes.append(obj_b)
            valid_frame_idx.append(j)

        valid_fnum = len(sub_bboxes)
        assert valid_fnum <= nframe, 'valid frame number should be smaller than nframe number'

        #interpolate new bbox
        if valid_fnum < nframe:
            sub_bboxes, obj_bboxes = interpolate(sub_bboxes, obj_bboxes, valid_frame_idx, sample_frames, nframe)

        sub_bboxes = {fid:bbox[:4].tolist() for fid, bbox in enumerate(sub_bboxes) }
        obj_bboxes = {fid:bbox[:4].tolist() for fid, bbox in enumerate(obj_bboxes) }

        # sub_bboxes = {fid: bbox[:4].tolist() for fid, bbox in enumerate(sub_bboxes) if bbox[4] >= beta_thresh}
        # obj_bboxes = {fid: bbox[:4].tolist() for fid, bbox in enumerate(obj_bboxes) if bbox[4] >= beta_thresh}

        video_res[relation]={"sub": sub_bboxes, "obj": obj_bboxes}

        pre_vname = vname

        if i == total_n -1:
            final_res[vname] = video_res

    save_results('results/ground_result.json', final_res)


def main():
    data_dir = '../ground_data/'
    dataset = 'vidvrd'
    val_list_file = 'dataset/{}/vrelation_val.json'.format(dataset)
    result_dir = '{}/results/{}'.format(data_dir, dataset)

    feat_dir = osp.join(data_dir, 'frame_feature')
    generate_track(val_list_file, result_dir, feat_dir)


if __name__ == "__main__":
    main()