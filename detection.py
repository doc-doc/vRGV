# ====================================================
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : detection.py
# ====================================================
from detect_frame import *
import dataloader
from tools.util import set_gpu_devices

BATCH_SIZE = 1
num_workers = 1
dataset = 'vidvrd/'
spatial_path = '../ground_data/'+dataset+'/JPEGImages/'
train_list_path= 'dataset/'+dataset+'/vrelation_train.json'
val_list_path = 'dataset/'+dataset+'/vrelation_val.json'


def main():

    data_loader = dataloader.DetectFrameLoader(BATCH_SIZE, num_workers, spatial_path,
                 dataset, train_list_path, val_list_path,
                 train_shuffle=False, val_shuffle=False)

    train_loader, val_loader = data_loader.run()

    checkpoint_path = 'models/pretrained_models/res101/coco/faster_rcnn_1_10_14657.pth'

    cfg_file = 'cfgs/res101_ls.yml'
    classes = ['coco']*81

    cuda = True
    class_agnostic = False
    set_gpu_devices(1)

    detect_frame = FeatureExtractor(train_loader, val_loader, cfg_file, classes,
                 class_agnostic, cuda, checkpoint_path)

    detect_frame.run()

if __name__ == "__main__":
    main()
