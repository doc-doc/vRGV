# ====================================================
# @Time    : 12/23/19 2:48 PM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : transfer_gt_annotation.py
# ====================================================
from util import load_file
import os.path as osp
import json


def transfer(gt_file, out_file):
    """
    merge discrete relations
    :param gt_file:
    :param out_file:
    :return:
    """
    gt = load_file(gt_file)
    gt_new = {}
    for vid, relations in gt.items():
        gt_new[vid] = {}
        for relation, sub_objs in relations.items():
            ins = []
            for sub_obj in sub_objs:
                duration = sub_obj['duration']
                sub = sub_obj['sub']
                obj = sub_obj['obj']
                new_sub = {}
                new_obj = {}
                s, e = duration[0], duration[1]
                for i in range(s, e):
                    new_sub[i] = sub[i-s]
                    new_obj[i] = obj[i-s]
                ins.append({"sub": new_sub, "obj": new_obj})
            gt_new[vid][relation]=ins

    with open(out_file, 'w') as fp:
        json.dump(gt_new, fp)



def main():

    dir = '../dataset/vidvrd/'
    gt_file = osp.join(dir, 'gt_relation.json')
    out_file = osp.join(dir, 'gt_relation_frame.json')

    transfer(gt_file, out_file)



if __name__ == "__main__":
    main()
