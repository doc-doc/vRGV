import os
import os.path as osp
import shutil
from util import load_file
import time

def delete_video(feature_dir, video_dir, video_list_file):
    video_list = load_file(video_list_file)
    for video in video_list:
        video_feature = osp.join(feature_dir, video)
        if not osp.exists(video_feature): continue
        mod_time = time.localtime(os.stat(video_feature).st_mtime)
        d = time.strftime('%d', mod_time)
        h = time.strftime('%H', mod_time)
        if int(d)<=10:
            video_frame = osp.join(video_dir, video)
            if not osp.exists(video_frame): continue
            shutil.rmtree(video_frame)
            print("Clean up {}".format(video_frame))


def main():
    dataset = 'vidor'
    data_dir = '/storage/jbxiao/workspace/ground_data/'+dataset
    anno_dir = '../dataset/'+dataset
    video_list_file = osp.join(anno_dir, 'train_list.json')
    
    feature_dir = osp.join(data_dir, 'features')
    video_dir = osp.join(data_dir, 'frames')

    delete_video(feature_dir, video_dir, video_list_file)

if __name__ == "__main__":
    main()
