# ====================================================
# @Time    : 11/15/19 3:06 PM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : relation2relation.py
# ====================================================
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import pickle as pkl


class AttHierarchicalGround(nn.Module):

    def __init__(self, input_size, hidden_size, visual_dim, word_dim, num_layers=1):
        super(AttHierarchicalGround, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_dim = hidden_size // 2


        self.num_layers = num_layers
        self.word_dim = word_dim

        self.max_seg_len = 12
        dropout = 0.5
        #self.max_seg_num = 10


        self.word_dict = None
        with open('/storage/jbxiao/workspace/ground_data/glove/vidvrd_word_glove.pkl', 'rb') as fp:
            self.word_dict = pkl.load(fp)

        self.embedding_word = nn.Sequential(nn.Linear(word_dim, self.embed_dim),
                                       nn.ReLU(),
                                       nn.Dropout(dropout))

        self.embedding_visual = nn.Sequential(nn.Linear(visual_dim, self.embed_dim),
                                       nn.ReLU(),
                                       nn.Dropout(dropout))

        # affine transformation for lstm hidden state
        # self.att_hw = nn.Linear(hidden_size, hidden_size)
        #
        # # affine transformation for context
        # self.att_cw = nn.Linear(hidden_size, hidden_size)
        #
        # # attention bias
        # self.att_bias = nn.Parameter(torch.zeros(hidden_size))
        #
        # # affine transformation for vector to scalar
        self.att_vec2sca = nn.Linear(hidden_size, 1)
        #
        # #self.temp_att = nn.linear(hidden_size*2, 1)
        self.transform_visual = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                       nn.ReLU(),
                                       nn.Dropout(dropout))

        self.transform_rel = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                       nn.ReLU(),
                                       nn.Dropout(dropout))

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.within_seg_rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.seg_rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)


    # def soft_attention(self, seq_num, input, context):
    #
    #     output = None
    #     for i in range(seq_num):
    #         y = input[: (i + 1)]
    #
    #         m = F.tanh(self.att_hw(y) + self.att_cw(context.expand(i + 1, -1, -1))
    #                    + self.att_bias.view(1, 1, -1).expand(i + 1, -1, -1))
    #
    #         m = self.att_vec2sca(m)
    #         s = F.softmax(m, dim=0)
    #
    #         z = torch.sum(y * s, dim=0).view(1, 1, -1)
    #
    #         if output is None:
    #             output = z
    #         else:
    #             output = torch.cat((output, z), dim=0)


    def word_embedding(self, word):
        """
        Extract GloVe feature for subject and object
        :param relation:
        :return:
        """
        word = word.lower()
        if word in self.word_dict.keys():
            word_embed = Variable(torch.from_numpy(np.asarray(self.word_dict[word],
                                                              dtype=np.float32))).squeeze().unsqueeze(0)
        else:
            word_embed = Variable(torch.zeros(1, self.word_dim))

        word_embed = self.embedding_word(word_embed.cuda())
        # word_embed = word_embed.cuda()

        return word_embed


    def attend_semantics(self, video, word):
        """
        attend subject and object in relation
        :param video:
        :param word:
        :return:
        """
        frame_count, nbbox, feature_dim = video.shape[0], video.shape[1], video.shape[2]
        video_embed = self.embedding_visual(video.view(-1, feature_dim))
        feature = video_embed.view(frame_count, nbbox, -1)
        #print(feature.shape) #(nframe, nbbox, feat_dim)

        attend_subject = video_embed.mm(word.view(word.shape[1], 1))
        attend_subject = attend_subject.view(frame_count, nbbox)
        attention = self.softmax(attend_subject)
        #print(attention.shape) #(nframe, nbbox)

        attend_feature = (feature * attention.unsqueeze(2)).sum(dim=1)
        #print(attend_feature.shape) #(nframe, feat_dim)

        return attend_feature, attention



    def spatialAtt(self, videos, relations):

        batch_size = videos.shape[0]
        frame_feat = None
        relation_feat = None

        for bs in range(batch_size):
            relation = relations[bs]
            video = videos[bs]

            split_relation = relation.split('-')
            subject, object = split_relation[0], split_relation[2]
            subject_embed = self.word_embedding(subject)
            object_embed = self.word_embedding(object)

            sub_obj = torch.cat([subject_embed, object_embed], dim=1).unsqueeze(0)



            subject_feat, sub_att = self.attend_semantics(video, subject_embed)
            object_feat, obj_att = self.attend_semantics(video, object_embed)

            cb_feat = torch.cat((subject_feat, object_feat), dim=1).unsqueeze(0)
            if bs == 0:
                frame_feat = cb_feat
                relation_feat = sub_obj
            else:
                frame_feat = torch.cat([frame_feat, cb_feat], 0)
                relation_feat = torch.cat([relation_feat, sub_obj], 0)


        # print(frame_feat.shape) #(batch_size, nframe, feat_dim)
        # print(relation_feat.shape)

        return frame_feat, relation_feat


    def temporalAtt(self, video, relation):

        # print(video.shape, relation.shape)

        att =self.att_vec2sca(self.relu(video + relation)) #(batch_size, nframe)
        temp_att = self.softmax(att)
        # print(temp_att.shape)

        att_feat = (video * temp_att) #(batch_size, nframe, feat_dim)

        # print(att_feat.shape)

        return att_feat



    def forward(self, videos, relation):

        frame_count = videos.shape[1]

        max_seg_num = frame_count / self.max_seg_len

        x, relation_feat = self.spatialAtt(videos, relation)

        x = self.transform_visual(x)

        # relation_feat = self.transform_rel(relation_feat)
        #
        att_x = self.temporalAtt(x, relation_feat)

        # x = torch.sum(videos, dim=2).squeeze()
        #
        # x = self.embedding_visual(x)

        within_seg_rnn_out, _ = self.within_seg_rnn(att_x)


        idx = np.round(np.linspace(self.max_seg_len-1, frame_count-1, max_seg_num))

        idx = [int(id) for id in idx]

        seg_rnn_input = within_seg_rnn_out[:,idx,:]

        # if idx[-1] != frame_count -1:
        #     seg_rnn_input = torch.cat((seg_rnn_input, within_seg_rnn_out[-1]))

        att_seg_rnn_input = self.temporalAtt(seg_rnn_input, relation_feat)

        # print(att_seg_rnn_input.shape)

        seg_out,(hn, cn) = self.seg_rnn(att_seg_rnn_input)

        # print(hn.shape, cn.shape)

        # output = self.soft_attention(frame_count, within_seg_rnn_out, context)
        output = (hn.permute(1, 0, 2), cn.permute(1, 0, 2))

        # print(output[0].shape, output[1].shape) #(batch_size, 1, feat_dim)

        return output


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=10):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length
        # self.init_weights()


    def init_weights(self):
        """
        Initialize some parameters with values from the uniform distribution
        :return:
        """
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)


    def forward(self, features, relations, lengths):
        """Decode relation attend video feature and reconstruct the relation."""
        embeddings = self.embed(relations)
        embeddings = torch.cat((features[0], embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])

        # print(outputs.shape)
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features[0]
        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)

            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
