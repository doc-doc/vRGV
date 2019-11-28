# ====================================================
# @Time    : 11/15/19 10:42 PM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : word_feature.py
# ====================================================
import numpy as np
import json
import os.path as osp
import pickle as pkl
from util import load_file




def word_selection(word_file, feature_file, out_file):
    """

    :param word_file:
    :param feature_file:
    :param out_file:
    :return:
    """
    with open(word_file, 'r') as fp:
        word_idx = json.load(fp)

    keys = []
    for key, item in word_idx.items():
        key = key.split('_')
        for k in key:
            keys.append(k)
    keys = set(keys)
    keys = sorted(list(keys))
    print(len(keys))
    print(keys)

    with open(feature_file, 'r') as fp:
        word_glove = fp.readlines()

    word_feature = {}
    for wg in word_glove:
        wg = wg.rstrip('\n').split(' ')
        if wg[0] in keys:
            word = wg[0]
            feature = np.asarray(wg[1:])
            print(feature.size)
            word_feature[word] = feature

    with open(out_file, 'wb') as fp:
        pkl.dump(word_feature, fp)

    print('finished')

def main():

    root_dir = '/storage/jbxiao/workspace/'
    word_file = osp.join(root_dir, 'ground_code/dataset/vidvrd/word_idx.json')

    feature_file = osp.join(root_dir, 'ground_data/glove/glove.6B.300d.txt')

    out_file = osp.join(root_dir, 'ground_data/glove/vidvrd_word_glove.pkl')

    word_selection(word_file, feature_file, out_file)


if __name__ == "__main__":
    main()