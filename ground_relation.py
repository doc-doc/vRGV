# ====================================================
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : ground_relation.py
# ====================================================
from networks.relation2relation import *
from utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os.path as osp
import time
from torch.nn.utils import clip_grad_value_


class GroundRelation():
    def __init__(self, vocab, train_loader, val_loader, checkpoint_path, model_prefix, vis_step, save_step, visual_dim, lr, batch_size, epoch_num, cuda):

        self.vocab = vocab
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_dir = checkpoint_path
        self.model_name = model_prefix
        self.vis_step = vis_step
        self.save_step = save_step

        self.lr = lr
        self.grad_clip = 10
        self.batch_size = batch_size
        self.epoch_num =  epoch_num
        self.cuda = cuda

        self.input_size = 512
        self.hidden_size = 512
        self.visual_dim = visual_dim
        self.word_dim = 300

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors


    def build_model(self):
        self.relation_ground = AttHierarchicalGround(self.input_size, self.hidden_size, self.visual_dim, self.word_dim)
        self.relation_reconstruction = DecoderRNN(self.input_size, self.hidden_size, len(self.vocab), 1, 10)

        params = [{'params':self.relation_reconstruction.parameters()}, {'params':self.relation_ground.parameters()}]
        self.optimizer = torch.optim.Adam(params=params,
                                             lr=self.lr)


        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=2, verbose=True)

        self.relation_ground.to(self.device)
        self.relation_reconstruction.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)


    def save_model(self, epoch):

        torch.save(self.relation_reconstruction.state_dict(),
                   osp.join(self.model_dir, '{}-reconstruct-{}.ckpt'.format(self.model_name, epoch + 1)))
        torch.save(self.relation_ground.state_dict(),
                   osp.join(self.model_dir, '{}-ground-{}.ckpt'.format(self.model_name, epoch + 1)))

    def resume(self):
        """
        initialize with pre-trained model
        :param epoch:
        :return:
        """
        ground_model_file = osp.join(self.model_dir,'../visual-ground-7.ckpt')
        reconstruct_model_file = osp.join(self.model_dir,'../visual-reconstruct-7.ckpt')
        ground_dict = torch.load(ground_model_file)
        reconstruct_dict = torch.load(reconstruct_model_file)

        new_ground_dict = {}
        for k, v in self.relation_ground.state_dict().items():
            if k in ground_dict:
                v = ground_dict[k]
            new_ground_dict[k] = v

        new_reconstruct_dict = {}
        for k, v in self.relation_reconstruction.state_dict().items():
            if k in reconstruct_dict:
                v = reconstruct_dict[k]
            new_reconstruct_dict[k] = v

        self.relation_reconstruction.load_state_dict(new_reconstruct_dict)


    def run(self, pretrain=False):

        self.build_model()
        if pretrain:
            self.resume()

        save_loss = np.inf

        for epoch in range(0, self.epoch_num):
            train_loss = self.train(epoch)
            val_loss = self.val(epoch)

            print('==> Epoch:[{}/{}] [Training loss: {:.4f} {:.4f} Val loss: {:.4f} {:.4f}]'.
                  format(epoch, self.epoch_num, train_loss, np.exp(train_loss), val_loss, np.exp(val_loss)))

            self.scheduler.step(val_loss)

            if val_loss < save_loss:
                save_loss = val_loss
                self.save_model(epoch)


    def train(self, epoch):
        print('==> Epoch:[{}/{}] [training stage Encode_lr: {} Decode_lr: {}]'.
              format(epoch, self.epoch_num, self.optimizer.param_groups[1]['lr'], self.optimizer.param_groups[0]['lr']))
        # print('Time \t Iter \t Loss \t Perplexity')

        self.relation_ground.train()
        self.relation_reconstruction.train()

        total_step = len(self.train_loader)
        epoch_loss = 0

        for iter, (relation_text, videos, relations, valid_lengths, video_names) in enumerate(self.train_loader):
            videos = videos.to(self.device)
            relations = relations.to(self.device)
            targets = pack_padded_sequence(relations, valid_lengths, batch_first=True)[0]

            video_out, video_hidden = self.relation_ground(videos, relation_text)

            relation_decode = self.relation_reconstruction(video_out, video_hidden, relations, valid_lengths)

            loss = self.criterion(relation_decode , targets)

            self.relation_ground.zero_grad()
            self.relation_reconstruction.zero_grad()

            loss.backward()

            # clip_gradient(self.optimizer, self.grad_clip)
            # clip_grad_value_()

            self.optimizer.step()

            cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            if iter % self.vis_step == 0:

                print('    [{}/{}]-{}-{:.4f}-{:5.4f}'.
                      format(iter, total_step, cur_time,  loss.item(), np.exp(loss.item())))

            epoch_loss += loss.item()

        return epoch_loss / total_step


    def val(self, epoch):
        print('==> Epoch:[{}/{}][validation stage]'.format(epoch, self.epoch_num))

        self.relation_ground.eval()
        self.relation_reconstruction.eval()

        total_step = len(self.val_loader)
        epoch_loss = 0

        for iter, (relation_text, videos, relations, valid_lengths, video_names) in enumerate(self.val_loader):
            videos = videos.to(self.device)
            relations = relations.to(self.device)
            targets = pack_padded_sequence(relations, valid_lengths, batch_first=True)[0]

            video_out, video_hidden = self.relation_ground(videos, relation_text)
            relation_decode = self.relation_reconstruction(video_out, video_hidden, relations, valid_lengths)

            loss = self.criterion(relation_decode, targets)

            cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            if iter % self.vis_step == 0:
                print('    [{}/{}]-{}-{:.4f}-{:5.4f}'.
                      format(iter, total_step, cur_time,  loss.item(), np.exp(loss.item())))


            epoch_loss += loss.item()

        return epoch_loss / total_step


    def predict(self, ep):

        self.build_model()

        ground_model_path = osp.join(self.model_dir, '{}-ground-{}.ckpt'.format(self.model_name, ep))
        reconstruction_path = osp.join(self.model_dir, '{}-reconstruct-{}.ckpt'.format(self.model_name, ep))

        self.relation_reconstruction.eval()
        self.relation_ground.eval()

        self.relation_ground.load_state_dict(torch.load(ground_model_path))
        self.relation_reconstruction.load_state_dict(torch.load(reconstruction_path))
        total = len(self.val_loader)
        pos_num = 0

        fout = open('results/prediction.txt', 'w')

        for iter, (relation_text, videos, relations, valid_lengths, video_names) in enumerate(self.val_loader):

            videos = videos.to(self.device)

            video_out, video_hidden = self.relation_ground(videos, relation_text)

            sample_ids = self.relation_reconstruction.sample(video_out, video_hidden)

            sample_ids = sample_ids[0].cpu().numpy()

            predict_relation = []
            for id in sample_ids:
                word = self.vocab.idx2word[id]
                predict_relation.append(word)
                if word == '<end>': break

            predict_relation = ' '.join(predict_relation[1:-1])
            # print(relation_text[0], predict_relation)

            table = str.maketrans('-_', '  ')
            relation = relation_text[0].translate(table)
            output_str = "{}:{}".format(relation, predict_relation)
            fout.writelines(output_str+'\n')

            if relation == predict_relation:
                pos_num += 1

            if iter%self.vis_step == 0:
                print("{}:{}".format(iter, output_str))

        print("Reconstrution Rate: ", pos_num / total)
        fout.close()


    def ground_attention(self, ep):
        """output the spatial temporal attention as grounding results"""
        self.build_model()
        ground_model_path = osp.join(self.model_dir, '{}-ground-{}.ckpt'.format(self.model_name, ep))
        self.relation_ground.eval()
        self.relation_ground.load_state_dict(torch.load(ground_model_path))

        total = len(self.val_loader)

        video = {}
        pre_vname = ''
        for iter, (relation_text, videos, relations, valid_lengths, video_names) in enumerate(self.val_loader):

            vname = video_names[0]
            videos = videos.to(self.device)
            sub_atts, obj_atts, beta1, beta2 = self.relation_ground(videos, relation_text, mode='val')

            data_sub_atts = sub_atts.data.cpu().numpy()
            data_obj_atts = obj_atts.data.cpu().numpy()
            data_beta2 = beta2.data.cpu().numpy()
            data_beta1 = beta1.data.cpu().numpy()

            data = {}
            data['sub'] = data_sub_atts.tolist()
            data['obj'] = data_obj_atts.tolist()
            data['beta2'] = data_beta2.tolist()
            data['beta1'] = data_beta1.tolist()


            if (vname != pre_vname and iter > 0) or (iter == total-1):
                if iter == total-1:
                    video[relation_text[0]] = data
                save_name = '../ground_data/results/vidvrd_new/'+pre_vname+'.json'
                save_results(save_name, video)
                video = {}

            video[relation_text[0]] = data

            pre_vname = vname

            if iter%self.vis_step == 0:
                print("{}:{}".format(iter, vname))
