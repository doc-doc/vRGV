# ====================================================
# @Time    : 11/19/19 2:34 PM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : get_missing_frame.py
# ====================================================
import os
import os.path as osp
from util import *
import numpy as np
import sys
import glob

def find_missing(anno_file, frame_file):
    annos = load_file(anno_file)
    vframes = {}
    for anno in annos:
        video, fnum = anno[0],anno[1]
        vframes[video] = fnum

    frames = load_file(frame_file)
    vframes_new = {}
    for f in frames:
        vname = f.split('/')[0]
        if vname in vframes_new:
            vframes_new[vname] += 1
        else:
            vframes_new[vname] = 1

    for vname, num in vframes.items():
        if vframes_new[vname] != num:
            print(vname, vframes_new[vname], num)


def fill_missing(anno_file, out_dir, video_dir):
    """

    :param anno_file:
    :param out_dir:
    :param video_dir:
    :return:
    """
    anno = load_file(anno_file)
    video_frame = {}
    for sample in anno:
        vname, nframe = sample[0], sample[1]
        video_frame[vname] = nframe
    sample_num = 512
    vid = 0
    # video_frame = sorted(video_frame.items(), key=lambda video_frame:video_frame[0])
    # for video in video_frame:
    #
    #     print(vid, video)
    #     vid += 1

    for vname, nframe in video_frame.items():
        if vid < 6971:
            vid += 1
            continue
        #if vid >=7000: break
        samples = np.round(np.linspace(
            1, nframe, sample_num))
        dst = osp.join(out_dir, vname)
        image_list = [osp.join(dst, '{:06d}.jpg'.format(int(sample))) for sample in samples]

        video = osp.join(video_dir, vname+'.mp4')
        extract_frames(video, dst)
        for fid in range(1, nframe+1):
            frame_name = osp.join(dst, str(fid).zfill(6)+'.jpg')
            if frame_name not in image_list:
                os.remove(frame_name)

        vid += 1
        # if vid % 25 == 0:
        print(vid, vname, nframe)

def find_mis_video(out_dir, video_dir):
    dirid = '1052'
    video_path = osp.join(video_dir, dirid)
    dst = osp.join(out_dir, dirid)
    videos = os.listdir(video_path)
    sample_num = 512

    for video in videos:

        bas_name = video.split('.')[0]
        if bas_name != '7546030642': continue

        vpath = osp.join(video_path, video)
        opath = osp.join(dst, bas_name)
        # if osp.exists(opath): continue
        extract_frames(vpath, opath)
        frames = os.listdir(opath)
        nframe = len(frames)

        samples = np.round(np.linspace(1, nframe, sample_num))
        samples = [int(s) for s in samples]
        for fid in range(1, nframe+1):
            if fid not in samples:
                frame_path = osp.join(opath, str(fid).zfill(6) + '.jpg')
                os.remove(frame_path)
        print(opath)
        break

def find_mis_all(out_dir, video_dir, anno_dir):
    dirid = '1110'
    video_path = osp.join(video_dir, dirid)
    dst = osp.join(out_dir, dirid)
    anno_path = osp.join(anno_dir, dirid)
    videos = os.listdir(anno_path)

    sample_num = 512
    frames = []

    for video in videos:
        # print(video)
        frames = []
        bas_name = video.split('.')[0]
        anno_file = osp.join(anno_path, '{}.json'.format(bas_name))
        with open(anno_file, 'r') as fp:
            anno = json.load(fp)
        fcount = anno['frame_count']
        vpath = osp.join(video_path, '{}.mp4'.format(bas_name))
        opath = osp.join(dst, bas_name)

        if osp.exists(opath):
            frames = os.listdir(opath)
        name = str(fcount).zfill(6)+'.jpg'

        if len(frames)>0:
            continue
            sort_frame = sorted(frames)
            final_name = sort_frame[-1]
            if final_name == name:
                continue

        extract_frames(vpath, opath)
        samples = np.round(np.linspace(1, fcount, sample_num))
        samples = [int(s) for s in samples]

        for fid in range(1, fcount+1):
            if fid not in samples:
                frame_path = osp.join(opath, str(fid).zfill(6) + '.jpg')
                os.remove(frame_path)
        print(opath)


def main():

    root_dir = '../dataset/vidor/'
    video_dir = '../../data/videos/'
    anno_file = osp.join(root_dir, 'vrelation_train.json')
    out_dir = '../../ground_data/vidor/frames/'
    anno_dir = '../../data/vidor-annotation/train/'
    # fill_missing(anno_file, out_dir, video_dir)
    # find_mis_video(out_dir, video_dir)
    find_mis_all(out_dir, video_dir, anno_dir)


if __name__ == "__main__":
    main()
