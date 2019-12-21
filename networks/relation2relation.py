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
        dropout = 0.2
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

        self.transform_spatt1 = nn.Linear(self.embed_dim*2, self.embed_dim)
        self.transform_spatt2 = nn.Linear(self.embed_dim, 1, bias=False)

        # affine transformation for lstm hidden state
        self.linear1 = nn.Linear(hidden_size*2, hidden_size)

        # affine transformation for context
        self.linear2 = nn.Linear(hidden_size, 1, bias=False)

        # attention bias
        # self.att_bias = nn.Parameter(torch.zeros(hidden_size))

        # # affine transformation for vector to scalar
        self.att_vec2sca = nn.Linear(hidden_size, 1, bias=False)
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


    def soft_attention(self, input, context):
        """
        compute temporal self-attention
        :param input:  (batch_size, seq_len, feat_dim)
        :param context: (batch_size, feat_dim)
        :return: vfeat: (batch_size, feat_dim)
        """
        batch_size, seq_len, feat_dim = input.size()
        context = context.unsqueeze(1).repeat(1, seq_len, 1)
        inputs = torch.cat((input, context), 2).view(-1, feat_dim*2)

        o = self.linear2(torch.tanh(self.linear1(inputs)))
        e = o.view(batch_size, seq_len)
        alpha = self.softmax(e)
        vfeat = torch.bmm(alpha.unsqueeze(1), input).squeeze(1)

        return vfeat



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
        video_embed = video_embed.view(frame_count, nbbox, -1)
        #print(feature.shape) #(nframe, nbbox, feat_dim)

        # attend_subject = video_embed.mm(word.view(word.shape[1], 1))
        # attend_subject = attend_subject.view(frame_count, nbbox)

        # attention = self.softmax(attend_subject)
        #print(attention.shape) #(nframe, nbbox)

        # attend_feature = (feature * attention.unsqueeze(2)).sum(dim=1)
        #print(attend_feature.shape) #(nframe, feat_dim)

        # return attend_feature, attention

        word = word.repeat(frame_count, nbbox, 1)

        video_word = torch.cat((video_embed, word), 2)

        video_word = video_word.view(frame_count*nbbox, -1)
        video_word_o = self.transform_spatt2(torch.tanh(self.transform_spatt1(video_word)))
        video_word_o = video_word_o.view(frame_count, nbbox)
        alpha = self.softmax(video_word_o)
        attend_feature = torch.bmm(alpha.unsqueeze(1), video_embed).squeeze(1)
        # print(alpha[0])
        # print(attend_feature.shape)
        return attend_feature, alpha



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

        att = self.att_vec2sca(self.relu(video + relation)) #(batch_size, nframe)
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

        # att_x = self.temporalAtt(x, relation_feat)

        within_seg_rnn_out, _ = self.within_seg_rnn(x)
        self.within_seg_rnn.flatten_parameters()

        idx = np.round(np.linspace(self.max_seg_len-1, frame_count-1, max_seg_num))

        idx = [int(id) for id in idx]

        seg_rnn_input = within_seg_rnn_out[:,idx,:]

        # if idx[-1] != frame_count -1:
        #     seg_rnn_input = torch.cat((seg_rnn_input, within_seg_rnn_out[-1]))

        # att_seg_rnn_input = self.temporalAtt(seg_rnn_input, relation_feat)

        seg_out, hidden = self.seg_rnn(seg_rnn_input)
        self.seg_rnn.flatten_parameters()

        output = self.soft_attention(within_seg_rnn_out, hidden[0].squeeze(0)) #(batch_size, feat_dim)
        # output = hidden[0].squeeze(0)

        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=10):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length
        # self._init_weights()


    def _init_weights(self):
        """
        Initialize some parameters with values from the uniform distribution
        :return:
        """
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)


    def _init_rnn_state(self, hidden):
        """
        initialize hidden state of decode with hidden state of encoder
        :param hidden:
        :return:
        """
        if hidden is None:
            return None
        if isinstance(hidden, tuple):
            hidden = tuple([self._cat_directions(h) for h in hidden])
        else:
            hidden = self._cat_directions(hidden)
        return hidden

    def _cat_directions(self, hidden):
        """
        if encoder is bi-directional, cat the bi-directional hidden state.
        (#directions * #layers, #batch, dim_hidden) -> (#layers, #batch, #directions * dim_hidden)
        :param hidden:
        :return:
        """
        hidden = torch.cat([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], 2)
        return hidden


    def forward(self, video_out, video_hidden, relations, lengths):
        """
        Decode relation attended video feature and reconstruct the relation.
        :param video_out: (batch, seq_len, dim_hidden * num_directions)
        :param video_hidden: (num_layers * num_directions, batch_size, dim_hidden)
        :param relations:
        :param lengths:
        :return:
        """
        embeddings = self.embed(relations)
        batch_size, seq_len, _ = embeddings.size()

        embeddings = torch.cat((video_out.unsqueeze(1), embeddings), 1)
        # print(embeddings.shape)

        # context = video_out.unsqueeze(1).repeat(1, seq_len, 1)

        # embeddings = torch.cat([embeddings, context], dim=2)

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

        hiddens, _ = self.lstm(packed, video_hidden)
        outputs = self.linear(hiddens[0])

        # print(outputs.shape)
        return outputs

    def sample(self, video_out, states=None):
        """reconstruct the relation using greedy search"""
        sampled_ids = []
        inputs = video_out.unsqueeze(1)
        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)

        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids

    # def sample(self, video_out, states=None):
    #     """reconstruct relation using greedy search.
    #     """
    #     batch_size, _ = video_out.size()
    #     sampled_ids = []
    #     start_id = torch.LongTensor([1] * batch_size).cuda() # word id 1 is start signal
    #     start_emded = self.embed(start_id)
    #
    #     # context = video_out.unsqueeze(1)
    #
    #     # inputs = torch.cat([start_emded.unsqueeze(1), context], dim=2)
    #     inputs = start_emded.unsqueeze(1)
    #     for i in range(self.max_seq_length):
    #
    #         hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
    #         outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
    #         _, predicted = outputs.max(1)  # predicted: (batch_size)
    #         sampled_ids.append(predicted)
    #         inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
    #         # inputs = torch.cat([inputs.unsqueeze(1), context], dim=2)  # inputs: (batch_size, 1, embed_size)
    #         inputs = inputs.unsqueeze(1)
    #     sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
    #     return sampled_ids
