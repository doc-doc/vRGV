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
        # if video == '1052/5441845281':
        #     print(video, nframe)
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

