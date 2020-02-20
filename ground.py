# ====================================================
# @Time    : 11/15/19 1:26 PM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : ground.py
# ====================================================
from ground_relation import *
import dataloader
from dataloader.build_vocab import Vocabulary
from tools.util import set_gpu_devices
import os.path as osp
import pickle as pkl


batch_size = 32
lr = 1e-4
num_workers = 10
epoch_num = 20
cuda = True
nframes, nbbox = 120, 40

vis_step = 30
save_step = 10000
visual_dim = 2048+5

dataset = 'vidvrd/'
root_dir = '/storage/jbxiao/workspace/'
video_feature_path = osp.join(root_dir, 'ground_data/{}/frame_feature/'.format(dataset))
video_feature_cache = osp.join(root_dir, 'ground_data/{}/video_feature'.format(dataset))

sample_list_path = osp.join('dataset/', dataset)
vocab_file = osp.join(sample_list_path, 'vocab.pkl')


checkpoint_path = osp.join('models', dataset)
model_prefix = 'visual_bbox_trans_temp2'

def main():

    with open(vocab_file, 'rb') as fp:
        vocab = pkl.load(fp)

    data_loader = dataloader.RelationLoader(batch_size, num_workers, video_feature_path, video_feature_cache,
                                            sample_list_path, vocab, nframes, nbbox, visual_dim, False, False)

    train_loader, val_loader = data_loader.run(mode='train')

    ground_relation = GroundRelation(vocab, train_loader, val_loader, checkpoint_path, model_prefix, vis_step, save_step, visual_dim,
                                     lr, batch_size, epoch_num, cuda)

    ground_relation.run(pretrain=False)
    # ground_relation.predict(6)
    # ground_relation.ground_attention(4)
    

if __name__ == "__main__":

    # set_gpu_devices(1)
    main()
