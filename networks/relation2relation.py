# ====================================================
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

        self.word_dict = None
        with open('/path/to/workspace/ground_data/glove/vidvrd_word_glove.pkl', 'rb') as fp:
            self.word_dict = pkl.load(fp)

        self.embedding_word = nn.Sequential(nn.Linear(word_dim, self.embed_dim),
                                       nn.ReLU(),
                                       nn.Dropout(dropout))

        self.embedding_visual = nn.Sequential(nn.Linear(visual_dim-5, self.embed_dim),
                                       nn.ReLU(),
                                       nn.Dropout(dropout))

        self.embedding_location = nn.Sequential(nn.Linear(5, self.embed_dim),
                                              nn.ReLU(),
                                              nn.Dropout(dropout))                           
        
        self.transform_spatt1 = nn.Linear(self.embed_dim*2, self.embed_dim)
        self.transform_spatt2 = nn.Linear(self.embed_dim, 1, bias=False)
       

        self.transform_tempatt_bottom1 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.transform_tempatt_bottom2 = nn.Linear(self.hidden_size, 1, bias=False)

       
        self.msg_sub2obj = nn.Sequential(nn.Linear(40, self.embed_dim),
                                       nn.ReLU(),
                                       nn.Dropout(dropout))

        self.msg_obj2sub = nn.Sequential(nn.Linear(40, self.embed_dim),
                                       nn.ReLU(),
                                       nn.Dropout(dropout))
       
        
        # affine transformation for lstm hidden state
        self.linear1 = nn.Linear(hidden_size*2, hidden_size)

        # affine transformation for context
        self.linear2 = nn.Linear(hidden_size, 1, bias=False)

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
        beta = self.softmax(e)
        vfeat = torch.bmm(beta.unsqueeze(1), input).squeeze(1)

        return vfeat, beta


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
            word_embed = Variable(torch.zeros(1, self.word_dim)) #will not go this branch

        word_embed = self.embedding_word(word_embed.cuda())

        return word_embed

    # def phrase_embedding(self, phrase):
    #     """
    #     average the word embeddings as phrase representation
    #     :param: phrase: eg, move_right
    #     """
    #     words = phrase.split('_')
    #     word_embed = Variable(torch.zeros(1, self.word_dim))
    #     for word in words:
    #         word_embed += self.word_embedding(word)
    #     phrase_embed = word_embed/len(words)
    #
    #     return  phrase_embed


    def attend_semantics(self, video_embed, word):
        """
        attend subject and object in relation
        :param video:
        :param word:
        :return:
        """

        frame_count, nbbox, feat_dim = video_embed.size()
        word = word.repeat(frame_count, nbbox, 1)

        video_word = torch.cat((video_embed, word), 2)

        video_word = video_word.view(frame_count*nbbox, -1)
        video_word_o = self.transform_spatt2(torch.tanh(self.transform_spatt1(video_word)))
        video_word_o = video_word_o.view(frame_count, nbbox)
        alpha = self.softmax(video_word_o) #(nframe, nbbox)
        attend_feature = torch.bmm(alpha.unsqueeze(1), video_embed).squeeze(1)

        return attend_feature, alpha


    def spatialAtt(self, videos, relations):

        frame_feat = None
        relation_feat = None
        sub_satt_values = None
        obj_satt_values = None

        batch_size, frame_count, nbbox, feat_dim = videos.size()
        videos = videos.view(-1, feat_dim)
        videos_visual = videos[:, :-5]
        videos_bbox = videos[:, -5:]
        visual_embeds = self.embedding_visual(videos_visual)
        bbox_embeds = self.embedding_location(videos_bbox)
        video_embeds = visual_embeds + bbox_embeds
        video_embeds = video_embeds.view(batch_size, frame_count, nbbox, -1)

        for bs in range(batch_size):
            relation = relations[bs]
            video_embed = video_embeds[bs]

            split_relation = relation.split('-')
            subject, object = split_relation[0], split_relation[2]
            subject_embed = self.word_embedding(subject)
            object_embed = self.word_embedding(object)
            # predicate_embed = self.phrase_embedding(split_relation[1])
            # sub_pred_obj = torch.cat([subject_embed, predicate_embed, object_embed], dim=1).unsqueeze(0)
            
            sub_obj = torch.cat([subject_embed, object_embed], dim=1).unsqueeze(0)
            #################for 2nd stage training#############################
            
            subject_feat, sub_att = self.attend_semantics(video_embed, subject_embed)
            object_feat, obj_att = self.attend_semantics(video_embed, object_embed)

            s2o_feat = self.msg_sub2obj(sub_att)
            o2s_feat = self.msg_obj2sub(obj_att)

            final_subject_feat = subject_feat + o2s_feat
            final_object_feat = object_feat + s2o_feat

            cb_feat = torch.cat((final_subject_feat, final_object_feat), dim=1).unsqueeze(0)

            sub_att, obj_att = sub_att.unsqueeze(0), obj_att.unsqueeze(0)

            if bs == 0:
                frame_feat = cb_feat
                relation_feat = sub_obj
                sub_satt_values = sub_att
                obj_satt_values = obj_att
            else:
                frame_feat = torch.cat([frame_feat, cb_feat], 0)
                relation_feat = torch.cat([relation_feat, sub_obj], 0)
                sub_satt_values = torch.cat([sub_satt_values, sub_att], 0)
                obj_satt_values = torch.cat([obj_satt_values, obj_att], 0)
            

        return frame_feat, relation_feat, sub_satt_values, obj_satt_values


    def temporalAtt(self, input, context):
        """
        compute temporal self-attention
        :param input:  (batch_size, seq_len, feat_dim)
        :param context: (batch_size, feat_dim)
        :return: vfeat: (batch_size, feat_dim)
        """
        # print(input.size(), context.size())

        batch_size, seq_len, feat_dim = input.size()
        # context = context.unsqueeze(1)
        context = context.repeat(1, seq_len, 1)
        inputs = torch.cat((input, context), 2).view(-1, feat_dim*2)

        o = self.transform_tempatt_bottom2(torch.tanh(self.transform_tempatt_bottom1(inputs)))
        e = o.view(batch_size, seq_len)
        beta = self.softmax(e)

        vfeat = beta.unsqueeze(2)* input
        # vfeat = torch.bmm(beta.unsqueeze(1), input)

        return vfeat, beta


    def forward(self, videos, relation_text, mode='train'):

        frame_count = videos.shape[1]

        max_seg_num = int(frame_count / self.max_seg_len)

        #SAU & MSG
        ori_x, ori_relation_feat, sub_atts, obj_atts = self.spatialAtt(videos, relation_text)
        # print(ori_x.shape, ori_relation_feat.shape)
        
        x_trans = self.transform_visual(ori_x)

        within_seg_rnn_out, _ = self.within_seg_rnn(x_trans)
        self.within_seg_rnn.flatten_parameters()

        idx = np.round(np.linspace(self.max_seg_len-1, frame_count-1, max_seg_num)).astype('int')

        seg_rnn_input = within_seg_rnn_out[:,idx,:]

        trans_relation_feat = self.transform_rel(ori_relation_feat)
        att_seg_rnn_input, beta2 = self.temporalAtt(seg_rnn_input, trans_relation_feat)
        

        seg_out, hidden = self.seg_rnn(att_seg_rnn_input)
        self.seg_rnn.flatten_parameters()

        output, beta1 = self.soft_attention(within_seg_rnn_out, hidden[0].squeeze(0)) #(batch_size, feat_dim)

        # output = hidden[0].squeeze(0)
        if mode != 'train':
            return sub_atts, obj_atts, beta1, beta2
        else:
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
