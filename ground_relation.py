# ====================================================
# @Time    : 11/15/19 1:30 PM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : ground_relation.py
# ====================================================
from networks.relation2relation import *
from utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os.path as osp
import time
# from progressbar import Percentage, Bar, ProgressBar, Timer

class GroundRelation():
    def __init__(self, vocab, train_loader, val_loader, checkpoint_path, vis_step, save_step, visual_dim, lr, batch_size, epoch_num, cuda):

        self.vocab = vocab
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_dir = checkpoint_path
        self.vis_step = vis_step
        self.save_step = save_step

        self.lr = lr
        self.grad_clip = 5
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
        # nn.utils.clip_grad_norm(params, 10.0)

        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=2, verbose=True)

        self.relation_ground.to(self.device)
        self.relation_reconstruction.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)


    def save_model(self, epoch):

        torch.save(self.relation_reconstruction.state_dict(),
                   osp.join(self.model_dir, 'reconstruction-{}.ckpt'.format(epoch + 1)))
        torch.save(self.relation_ground.state_dict(),
                   osp.join(self.model_dir, 'relation_ground-{}.ckpt'.format(epoch + 1)))


    def run(self):

        self.build_model()

        save_loss = 10000

        for epoch in range(self.epoch_num):
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

        for iter, (relation_text, videos, relations, valid_lengths) in enumerate(self.train_loader):
            videos = videos.to(self.device)
            relations = relations.to(self.device)
            targets = pack_padded_sequence(relations, valid_lengths, batch_first=True)[0]

            video_encode = self.relation_ground(videos, relation_text)

            relation_decode = self.relation_reconstruction(video_encode, relations, valid_lengths)

            loss = self.criterion(relation_decode , targets)




            self.relation_ground.zero_grad()
            self.relation_reconstruction.zero_grad()

            loss.backward()

            clip_gradient(self.optimizer, self.grad_clip)

            self.optimizer.step()

            cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            if iter % self.vis_step == 0:
                # print(video_encode[0][0, 0][:30])
                print(torch.argmax(relation_decode, 1))
                print(targets)
                print(relation_text)

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

        for iter, (relation_text, videos, relations, valid_lengths) in enumerate(self.val_loader):
            videos = videos.to(self.device)
            relations = relations.to(self.device)
            targets = pack_padded_sequence(relations, valid_lengths, batch_first=True)[0]

            video_encode = self.relation_ground(videos, relation_text)
            relation_decode = self.relation_reconstruction(video_encode, relations, valid_lengths)

            loss = self.criterion(relation_decode, targets)

            cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            if iter % self.vis_step == 0:
                print('    [{}/{}]-{}-{:.4f}-{:5.4f}'.
                      format(iter, total_step, cur_time,  loss.item(), np.exp(loss.item())))


            epoch_loss += loss.item()

        return epoch_loss / total_step


    def predict(self, ep):

        self.build_model()

        ground_model_path = osp.join(self.model_dir, 'relation_ground-{}.ckpt'.format(ep))
        reconstruction_path = osp.join(self.model_dir, 'reconstruction-{}.ckpt'.format(ep))

        self.relation_reconstruction.eval()
        self.relation_ground.eval()

        self.relation_ground.load_state_dict(torch.load(ground_model_path))
        self.relation_reconstruction.load_state_dict(torch.load(reconstruction_path))

        for iter, (relation_text, videos, relations, valid_lengths) in enumerate(self.val_loader):

            videos = videos.to(self.device)

            video_encode = self.relation_ground(videos, relation_text)

            # print(video_encode[0][0,0][:30])

            sample_ids = self.relation_reconstruction.sample(video_encode)

            sample_ids = sample_ids[0].cpu().numpy()

            predict_relation = []
            for id in sample_ids:
                word = self.vocab.idx2word[id]
                predict_relation.append(word)
                if word == '<end>': break

            predict_relation = ' '.join(predict_relation)
            print(relation_text[0], predict_relation)




