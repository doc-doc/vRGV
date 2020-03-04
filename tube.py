# ====================================================
# @Time    : 26/2/20 10:47 AM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : tube.py
# ====================================================
import numpy as np
import copy

def remove_dup_frame(video_bboxes, sample_frames):
    valid_frame_idx = []
    new_video_bbox = []
    for j, bbox in enumerate(video_bboxes):
        if j > 0 and sample_frames[j] == sample_frames[j-1]: continue
        new_video_bbox.append(bbox)
        valid_frame_idx.append(j)

    return new_video_bbox, valid_frame_idx


def find_continus(num_list, sample_rate):
    """
    find the continue numbers in list
    :param num_list:
    :return:
    """
    res_lists = []
    s = 1
    n = len(num_list)
    max_sp_step = 10
    min_len = int(max_sp_step * sample_rate)
    margine = int(max_sp_step * sample_rate)

    while s < n:
        if num_list[s]-num_list[s-1] <= margine:
            flag = s - 1
            while num_list[s]-num_list[s-1] <= margine:
                s += 1
                if s >= n: break
            if s-flag <= min_len: continue
            res_lists.append(num_list[flag:s])
        else:
            s += 1

    return res_lists


def temporal_ground(temp, nframe, t_thresh):
    """

    :param temp:
    :param nframe:
    :return:
    """
    sample_nframe = len(temp)
    sample_rate = sample_nframe/nframe
    temp_inds = np.where(temp>=t_thresh)[0]
    clips = find_continus(temp_inds, sample_rate)

    return clips, sample_rate


def get_area(bbox):
     area = (bbox[2]-bbox[0]+1)*(bbox[3]-bbox[1]+1)
     return area


def build_link(bboxes, scores_sub, scores_obj, clip, sample_frames):
    """
    build link between adjacent frames
    :param bboxes:
    :param scores:
    :param clip:
    :param sample_frames:
    :return:
    """

    link_scores_sub = []
    link_scores_obj = []
    cbboxes = []
    valid_frame_idx = []

    test_samples = []

    for idx, fid in enumerate(clip):
        test_samples.append(sample_frames[fid])
    csample_frames = sorted(list(set(test_samples)))
    valid_fnum = len(csample_frames)

    for id in range(len(clip)):
        if id == len(clip)-1:
            cbboxes.append(bboxes[clip[id]])
            valid_frame_idx.append(id)
            break
        real_gap = sample_frames[clip[id+1]] - sample_frames[clip[id]]
        if real_gap == 0:
            continue

        factor = 1/(real_gap) #0.5 #0.5 * (1 / real_gap)
        valid_frame_idx.append(id)

        bboxes_1, bboxes_2 = bboxes[clip[id]], bboxes[clip[id+1]]
        sub_scores_1, sub_scores_2 = scores_sub[clip[id]], scores_sub[clip[id+1]]
        obj_scores_1, obj_scores_2 = scores_obj[clip[id]], scores_obj[clip[id+1]]


        areas_1 = np.array([get_area(bbox) for bbox in bboxes_1])

        areas_2 = np.array([get_area(bbox) for bbox in bboxes_2])
        sub_link_frame = []
        obj_link_frame = []
        for bid, bbox in enumerate(bboxes_1):
            area_1 = areas_1[bid]
            sub_score_1 = sub_scores_1[bid]
            obj_score_1 = obj_scores_1[bid]
            x1 = np.maximum(bbox[0], bboxes_2[:, 0])
            y1 = np.maximum(bbox[1], bboxes_2[:, 1])
            x2 = np.minimum(bbox[2], bboxes_2[:, 2])
            y2 = np.minimum(bbox[3], bboxes_2[:, 3])
            W = np.maximum(0, x2 - x1 + 1)
            H = np.maximum(0, y2 - y1 + 1)
            ov_area = W * H
            IoUs = ov_area / (area_1 + areas_2 - ov_area) # IoUs betweem a bbox in frame t and all bboxes in frame t+1
            sub_cur_scores = sub_score_1 + sub_scores_2 + factor*IoUs  # link scores between a bbox in frame t and all bboxes in frame t+1
            obj_cur_scores = obj_score_1 + obj_scores_2 + factor*IoUs
            sub_link_frame.append(sub_cur_scores)
            obj_link_frame.append(obj_cur_scores)
        link_scores_sub.append(sub_link_frame)
        link_scores_obj.append(obj_link_frame)
        cbboxes.append(bboxes[clip[id]])

    assert len(link_scores_sub)==valid_fnum-1, 'remove redundant frame error {} v.s {}'.format(len(link_scores_sub), valid_fnum-1)

    return np.asarray(link_scores_sub), np.asarray(link_scores_obj), np.asarray(cbboxes), valid_frame_idx


def find_max_path(link_scores, init_score):
    """
    viterbi algorithm
    s(o, k) = max{s(o,j)+w(j,k)} j nodes in stage i, k nodes in stage i+1
    find the path of maximal score
    :param link_scores: (stage_num-1)*40*40
    :return:
    """
    stage_num = link_scores.shape[0]
    node_per_stage = link_scores.shape[1]
    max_path_score = np.zeros((stage_num, node_per_stage))
    path = np.zeros((node_per_stage, stage_num), int)

    for i in range(node_per_stage):
        max_path_score[0][i] = init_score[i]
        path[i][0] = i

    for i in range(1, stage_num):
        #enumerate nodes in stage (i)
        newpath = np.zeros((node_per_stage, stage_num), int)
        for j in range(node_per_stage):
            # enumerate nodes in stage (i+1)
            prob = -1
            for k in range(node_per_stage):
                link_score = link_scores[i-1][j][k]
                temp = max_path_score[i-1][j] + link_score
                if temp > max_path_score[i][k]:
                    max_path_score[i][k] = temp
                    for l in range(i):
                        newpath[k][l] = path[j][l]
                    newpath[k][i] = k

        path = newpath

    max_score = 0
    path_state = 0
    for j in range(node_per_stage):
        if max_path_score[stage_num-1][j] > max_score:
            max_score = max_path_score[stage_num-1][j]
            path_state = j

    # print(max_score)
    # print(path[path_state])

    return path[path_state], max_score


def get_tube(bboxes, alpha_s, alpha_o, clip, sample_frames):
    """

    :param bboxes:
    :param scores:
    :param clip:
    :param sample_frames:
    :param sample_rate:
    :return:
    """
    # alpha_s= np.sort(alpha_s, 1)
    # alpha_o = np.sort(alpha_o, 1)
    link_scores_sub, link_scores_obj, cbboxes, valid_frame_idx = build_link(bboxes, alpha_s, alpha_o, clip, sample_frames)
    maxpath_sub, max_score_sub = find_max_path(link_scores_sub, alpha_s[clip[valid_frame_idx[0]]])
    maxpath_obj, max_score_obj = find_max_path(link_scores_obj, alpha_o[clip[valid_frame_idx[0]]])

    tube_len = cbboxes.shape[0]

    assert tube_len-1 == len(maxpath_sub), 'error linked path'
    sub_tube, obj_tube = [],[]
    for fid in range(tube_len-1):
            sub_id = maxpath_sub[fid]
            obj_id = maxpath_obj[fid]
            sub_tube.append(cbboxes[fid][sub_id].tolist())
            obj_tube.append(cbboxes[fid][obj_id].tolist())

    s_idx = np.argmax(alpha_s[clip[valid_frame_idx[-1]]])
    o_idx = np.argmax(alpha_o[clip[valid_frame_idx[-1]]])
    sub_tube.append(cbboxes[-1][s_idx].tolist())
    obj_tube.append(cbboxes[-1][o_idx].tolist())
    max_score_sub /= tube_len
    max_score_obj /= tube_len


    return sub_tube, max_score_sub, obj_tube, max_score_obj,valid_frame_idx



def link_bbox(video_bboxes, alpha_s, alpha_o, temp, beta_thresh, sample_frames, nframe):
    """

    :param video_bboxes: T*40*4
    :param alpha_s: spatial attention for subject  40*T
    :param alpha_o: spatial attention for object 40*T
    :param temp: temporal attention 1*T
    :param sample_frames: real sampled frame id
    :return:
    """
    clips, sample_rate = temporal_ground(temp, nframe, beta_thresh)
    T_s, T_o, sid = None, None, 0
    max_score = 0
    valid_frame_idx = None

    for clip in clips:
        sub, sub_score, obj, obj_score, clip_frame_idx = get_tube(video_bboxes, alpha_s, alpha_o, clip, sample_frames)
        pair_score = sub_score+obj_score
        if pair_score > max_score:
            T_s = sub
            T_o = obj
            max_score = pair_score
            valid_frame_idx = clip_frame_idx
            sid = clip[valid_frame_idx[0]]

    return T_s, T_o, sid, valid_frame_idx










