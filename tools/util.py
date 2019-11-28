# ====================================================
# @Time    : 11/7/19 7:11 PM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : util.py
# ====================================================
import json
import os.path as osp
import os

def load_file(file_name):

    with open(file_name, 'r') as fp:
        if osp.splitext(file_name)[1]== '.txt':
            annos = fp.readlines()
            annos = [line.rstrip() for line in annos]
        if osp.splitext(file_name)[1] == '.json':
            annos = json.load(fp)

    return annos

def set_gpu_devices(gpu_id):
    gpu = ''
    if gpu_id != -1:
        gpu = str(gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
