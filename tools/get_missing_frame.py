# ====================================================
# @Time    : 11/19/19 2:34 PM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : get_missing_frame.py
# ====================================================
import os.path as osp
from util import load_file


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




def main():

    root_dir = '../dataset/vidvrd/'
    frame_file = osp.join(root_dir, 'test.txt')
    anno_file = osp.join(root_dir, 'vrelation_val.json')
    find_missing(anno_file, frame_file)


if __name__ == "__main__":
    main()
