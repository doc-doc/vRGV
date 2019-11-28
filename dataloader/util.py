# ====================================================
# @Time    : 11/14/19 10:53 PM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : util.py
# ====================================================
import json
import os
import os.path as osp


def load_file(file_name):

    with open(file_name, 'r') as fp:
        if osp.splitext(file_name)[1]== '.txt':
            annos = fp.readlines()
            annos = [line.rstrip() for line in annos]
        if osp.splitext(file_name)[1] == '.json':
            annos = json.load(fp)

    return annos





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
    for video, nframe in vframes.items():
        fnames = [osp.join(video, str(fid).zfill(6)) for fid in range(nframe)]
        if all_frames == []:
            all_frames = fnames
        else:
            all_frames.extend(fnames)

    return all_frames

