# ====================================================
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : util.py
# ====================================================
import os.path as osp
import json


def load_file(file_name):

    with open(file_name, 'r') as fp:
        if osp.splitext(file_name)[1]== '.txt':
            annos = fp.readlines()
            annos = [line.rstrip() for line in annos]
        if osp.splitext(file_name)[1] == '.json':
            annos = json.load(fp)

    return annos


def iou(bbox_1, bbox_2):
    """
    Get IoU value of two bboxes
    :param bbox_1:
    :param bbox_2:
    :return: IoU
    """
    w_1 = bbox_1[2] - bbox_1[0] + 1
    h_1 = bbox_1[3] - bbox_1[1] + 1
    w_2 = bbox_2[2] - bbox_2[0] + 1
    h_2 = bbox_2[3] - bbox_2[1] + 1
    area_1 = w_1 * h_1
    area_2 = w_2 * h_2

    overlap_bbox = (max(bbox_1[0], bbox_2[0]), max(bbox_1[1], bbox_2[1]),
                    min(bbox_1[2], bbox_2[2]), min(bbox_1[3], bbox_2[3]))
    overlap_w = max(0, (overlap_bbox[2] - overlap_bbox[0] + 1))
    overlap_h = max(0, (overlap_bbox[3] - overlap_bbox[1] + 1))

    overlap_area = overlap_w * overlap_h
    union_area = area_1 + area_2 - overlap_area
    IoU = overlap_area * 1.0 / union_area
    return IoU
