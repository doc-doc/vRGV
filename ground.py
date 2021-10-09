# ====================================================
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : ground.py
# ====================================================
from ground_relation import *
from dataloader.ground_loader import *
from dataloader.build_vocab import Vocabulary
import os.path as osp
import pickle as pkl
from argparse import ArgumentParser


batch_size = 32
lr = 1e-4
num_workers = 8
epoch_num = 10
cuda = True
nframes, nbbox = 120, 40

vis_step = 200
save_step = 10000
visual_dim = 2048+5 #visual appearance+bbox

dataset = 'vidvrd/'
root_dir = '/path/to/workspace/' #this directory includes two folders: ground_data and vRGV

video_feature_path = osp.join(root_dir, 'ground_data/{}/frame_feature/'.format(dataset))
video_feature_cache = osp.join(root_dir, 'ground_data/{}/video_feature/'.format(dataset))

sample_list_path = osp.join('dataset/', dataset)
vocab_file = osp.join(sample_list_path, 'vocab.pkl')

checkpoint_path = osp.join('models', dataset)
model_prefix = 'visual_bbox_trans_temp2'


def main(args):

    with open(vocab_file, 'rb') as fp:
        vocab = pkl.load(fp)

    data_loader = RelationLoader(batch_size, num_workers, video_feature_path, video_feature_cache,
                                            sample_list_path, vocab, nframes, nbbox, visual_dim, True, False)

    train_loader, val_loader = data_loader.run(mode=args.mode)

    ground_relation = GroundRelation(vocab, train_loader, val_loader, checkpoint_path, model_prefix, vis_step, save_step, visual_dim,
                                     lr, batch_size, epoch_num, cuda)

    mode = args.mode
    if mode == 'train':
        ground_relation.run(pretrain=False)
    elif mode == 'val':
        #return relation-aware spatio-temporal attention for dynamicly linking object proposals into trajectories
        save_name = '../ground_data/results/vidvrd_batch.json'
        ground_relation.ground_attention(7, save_name)
    

if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    torch.backends.cudnn.benchmark = True

    parser = ArgumentParser()
    parser.add_argument('--mode', dest='mode', type=str, default='train', help='train or val')
    args = parser.parse_args()
    main(args)
