# ====================================================
# @Time    : 11/7/19 4:34 PM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : data_analysis.py
# ====================================================

import numpy as np
from util import *
import matplotlib.pyplot as plt
import pickle as pkl

work_dir = '/storage/jbxiao/workspace/'


def parse(video_files, anno_dir):
    video_names = load_file(video_files)

    durations = []
    for vname in video_names:
        if '.json' not in vname:
            vname = vname.rstrip('\n')+'.json'
        name = osp.join(anno_dir, vname)
        with open(name, 'r') as fp:
            anno = json.load(fp)
        relations = anno['relation_instances']
        for relation in relations:
            begin_fid = relation['begin_fid']
            end_fid = relation['end_fid']
            duration = end_fid - begin_fid
            durations.append(duration)

    relation_num = len(durations)
    print(relation_num)
    return np.asarray(durations)


def main():
    # data_dir = osp.join(work_dir, 'vidor-dataset/')
    # anno_dir = osp.join(data_dir,'annotation/train')
    # video_file = osp.join(data_dir, 'train_list.json')
    # data_dir = osp.join(work_dir, 'vdata')
    # anno_dir = osp.join(data_dir, 'vidvrd')
    # video_file = osp.join(data_dir, 'val_list.txt')
    # durations = parse(video_file, anno_dir)
    # with open('durations.pkl', 'wb') as fp:
    #     pkl.dump(durations, fp)
    with open('durations.pkl', 'rb') as fp:
        durations = pkl.load(fp)
    freq = np.bincount(durations)
    print(freq)
    freq_num = np.argmax(freq)
    print(durations.min(), durations.max(), durations.mean(), np.median(durations), freq_num)

    plt.figure()
    total = sum(freq)
    print(total)
    print(len(freq))
    num = 90

    print('<3s: ', sum(freq[:num]))
    print('<5s:', sum(freq[:160]))

    plt.bar(range(0,num),np.log(freq[:num]))
    plt.show()


if __name__ == "__main__":
    main()
