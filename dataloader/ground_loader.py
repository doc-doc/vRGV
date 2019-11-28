# ====================================================
# @Time    : 11/14/19 10:42 PM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : ground_loader.py
# ====================================================
import torch
from torch.utils.data import Dataset, DataLoader
from .util import load_file
import os.path as osp
import pickle as pkl
import numpy as np
import nltk
import string

class RelationDataset(Dataset):
    """load the dataset in dataloader"""

    def __init__(self, video_feature_path, sample_list_path, vocab, mode, nframes, nbbox, visual_dim):
        self.video_feature_path = video_feature_path
        self.vocab = vocab
        self.frame_steps = nframes
        self.nbbox = nbbox
        sample_list_file = osp.join(sample_list_path, 'vrelation_{}.json'.format(mode))
        self.sample_list = load_file(sample_list_file)
        self.feat_dim = visual_dim


    def __len__(self):
        return len(self.sample_list)

    def get_video_feature(self, video_name, frame_count, width, height):
        """

        :param video_name:
        :param frame_count:
        :param width:
        :param height:
        :return:
        """
        video_feature_folder = osp.join(self.video_feature_path, video_name)


        sample_frames = np.round(np.linspace(0, frame_count - 1, self.frame_steps))


        video_feature = torch.zeros(len(sample_frames), self.nbbox, self.feat_dim)
        for i, fid in enumerate(sample_frames):
            frame_name = osp.join(video_feature_folder, str(int(fid)).zfill(6)+'.pkl')
            # bbox, classme, feat = load_predict(frame_name, self.nbbox)
            with open(frame_name, 'rb') as fp:
                feat = pkl.load(fp)['cls_prob']
            video_feature[i] = torch.Tensor(feat)

        # print(video_feature[10])

        return video_feature


    def get_word_idx(self, relation):
        """
        convert relation to index sequence
        :param relation:
        :return:
        """
        table = str.maketrans('-_', '  ')
        relation = relation.translate(table)

        tokens = nltk.tokenize.word_tokenize(str(relation).lower())
        relation = []
        relation.append(self.vocab('<start>'))
        relation.extend([self.vocab(token) for token in tokens])
        relation.append(self.vocab('<end>'))
        target = torch.Tensor(relation)

        return target


    def __getitem__(self, idx):
        """
        return an item from data list as tuple (video, relation)
        :param idx:
        :return:
                -video: torch tensor (nframe, nbbox, feat)
                -relation: torch tensor of variable length
        """
        video_name, frame_count, width, height, relation = self.sample_list[idx]
        video_feature = self.get_video_feature(video_name, frame_count, width, height)
        relation2idx = self.get_word_idx(relation)
        return video_feature, relation2idx, relation


class RelationLoader():
    def __init__(self, batch_size, num_worker, video_feature_path,
                 sample_list_path, vocab, nframes, nbbox, visual_dim, train_shuffle=True, val_shuffle=False):
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.video_feature_path = video_feature_path
        self.sample_list_path = sample_list_path
        self.vocab = vocab
        self.nframes = nframes
        self.nbbox = nbbox
        self.visual_dim = visual_dim
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle


    def run(self):
        train_loader = self.train()
        val_loader = self.validate()
        return train_loader, val_loader

    def train(self):
        # print("Now in train")
        # applying trabsformation on training videos
        training_set = RelationDataset(self.video_feature_path, self.sample_list_path,
                                       self.vocab, 'train', self.nframes, self.nbbox,
                                       self.visual_dim)

        print('Eligible video-relation pairs for training : {}'.format(len(training_set)))
        train_loader = DataLoader(
            dataset=training_set,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_worker,
            collate_fn=collate_fn)

        return train_loader

    def validate(self):
        # print("Now in Validate")
        # applying transformation for validation videos
        validation_set = RelationDataset(self.video_feature_path, self.sample_list_path,
                                         self.vocab, 'val', self.nframes, self.nbbox,
                                         self.visual_dim)

        print('Eligible video-relation pairs for validation : {}'.format(len(validation_set)))
        val_loader = DataLoader(
            dataset=validation_set,
            batch_size=self.batch_size,
            shuffle=self.val_shuffle,
            num_workers=self.num_worker,
            collate_fn=collate_fn)

        return val_loader


def collate_fn (data):
    """
    Create mini-batch tensors from the list of tuples (video, relation)
    :param data:
                -video: torch tensor of shape (nframe, nbbox, feat_dim)
                -relation2idx: torch tensor of variable length
                -relation: raw relation
    :return:
        images: torch tensor of shape (batch_size, nframe, nbbox, feat_dim)
        targets: torch tensor of shape (batch_size, padded_length)
        length: valid length for each padded length
    """
    data.sort(key=lambda x : len(x[1]), reverse=True)
    videos, relation2idx, relations= zip(*data)

    #merge videos
    videos = torch.stack(videos, 0)

    #merge relations
    lengths = [len(rel) for rel in relation2idx]
    targets = torch.zeros(len(relation2idx), max(lengths)).long()
    for i, rel in enumerate(relation2idx):
        end = lengths[i]
        targets[i, :end] = rel[:end]

    return relations, videos, targets, lengths
