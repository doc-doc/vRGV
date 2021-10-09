# ====================================================
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : basic.py
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

        self.embedding_word = nn.Sequential(nn.Linear(word_dim, self.embed_dim),
                                       nn.ReLU(),
                                       nn.Dropout(dropout))

        self.embedding_visual = nn.Sequential(nn.Linear(visual_dim, hidden_size),
                                       nn.ReLU(),
                                       nn.Dropout(dropout))
        
        # affine transformation for lstm hidden state
        self.linear1 = nn.Linear(hidden_size*2, hidden_size)

        # affine transformation for context
        self.linear2 = nn.Linear(hidden_size, 1, bias=False)

        self.transform_visual = nn.Sequential(nn.Linear(hidden_size, hidden_size),
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
        :return: vfeat: (batch_size, feat_dim), beta
        """
        batch_size, seq_len, feat_dim = input.size()
        context = context.unsqueeze(1).repeat(1, seq_len, 1)
        inputs = torch.cat((input, context), 2).view(-1, feat_dim*2)

        o = self.linear2(torch.tanh(self.linear1(inputs)))
        e = o.view(batch_size, seq_len)
        beta = self.softmax(e)
        vfeat = torch.bmm(beta.unsqueeze(1), input).squeeze(1)

        return vfeat, beta


    def forward(self, videos, relation_text, mode='train'):
        """
        Without participation of textual relation, to warm-up the decoder only
        """

        frame_count = videos.shape[1]

        max_seg_num = int(frame_count / self.max_seg_len)

        ori_x = self.embedding_visual(videos).sum(dim=2).squeeze()
        
        x_trans = self.transform_visual(ori_x)

        within_seg_rnn_out, _ = self.within_seg_rnn(x_trans)
        self.within_seg_rnn.flatten_parameters()

        idx = np.round(np.linspace(self.max_seg_len-1, frame_count-1, max_seg_num)).astype('int')

        seg_rnn_input = within_seg_rnn_out[:,idx,:]
        
        seg_out, hidden = self.seg_rnn(seg_rnn_input)
        self.seg_rnn.flatten_parameters()
        
        output, _ = self.soft_attention(within_seg_rnn_out, hidden[0].squeeze(0))

        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=10):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length

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
