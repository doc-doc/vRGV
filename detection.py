# ====================================================
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : detection.py
# ====================================================
from detect_frame import *
from dataloader.detect_frame_loader import *
from tools.util import set_gpu_devices
from argparse import ArgumentParser

BATCH_SIZE = 1
num_workers = 1
dataset = 'vidvrd/'
spatial_path = '../ground_data/'+dataset+'/JPEGImages/'
train_list_path= 'dataset/'+dataset+'/vrelation_train.json'
val_list_path = 'dataset/'+dataset+'/vrelation_val.json'


def main(args):

    data_loader = DetectFrameLoader(BATCH_SIZE, num_workers, spatial_path,
                 dataset, train_list_path, val_list_path,
                 train_shuffle=False, val_shuffle=False)

    train_loader, val_loader = data_loader.run(args.mode)

    checkpoint_path = 'models/pretrained_models/res101/coco/faster_rcnn_1_10_14657.pth'
    save_dir = '../ground_data/vidvrd/frame_feature1/'

    cfg_file = 'cfgs/res101_ls.yml'
    classes = ['coco']*81

    cuda = True
    class_agnostic = False

    detect_frame = FeatureExtractor(train_loader, val_loader, cfg_file, classes,
                 class_agnostic, cuda, checkpoint_path, save_dir)

    detect_frame.run(args.mode)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--mode', dest='mode', type=str, default='val', help='train or val')
    args = parser.parse_args()
    main(args)

