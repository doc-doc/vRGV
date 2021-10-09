import sys
sys.path.insert(0, '../')
from dataloader.util import *

def sample_video_feature(src, dst, sample_list_file):
    nframes, nbbox, feat_dim = 120, 40, 2048+5
    samples = load_file(sample_list_file)
    sp_num = len(samples)
    for it, item in enumerate(samples):
        # if it < 24000: continue
        # if it >= 24000: break
        video_name, frame_count, width, height, relation = item
        dst_file = osp.join(dst, video_name)
        src_dir = osp.join(src, video_name)
        if osp.exists(dst_file+'.npy'): 
            print('exist {}.npy'.format(dst_file))
            continue
        get_video_feature(src_dir, dst_file, frame_count, width, height, nbbox, nframes, feat_dim)
        if it % 200 == 0:
            print(it, sp_num)


def main():
    dataset = 'vidvrd/'
    root_dir = '/path/to/workspace/' #this directory includes two folders: ground_data and vRGV
    video_feature_path = osp.join(root_dir, 'ground_data/{}/frame_feature/'.format(dataset))
    video_feature_cache = osp.join(root_dir, 'ground_data/{}/video_feature/'.format(dataset))
    dset_dir = '../dataset/'+dataset+'/'
    train_list = osp.join(dset_dir, 'vrelation_train.json')
    val_list = osp.join(dset_dir, 'vrelation_val.json')
    sample_video_feature(video_feature_path, video_feature_cache, val_list)
    # sample_video_feature(video_feature_path, video_feature_cache, train_list)


if __name__ == "__main__":
    main()
