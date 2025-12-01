import torch
import torch.nn as nn
import numpy as np
from attention import Attention,Transformer_encoder
from ABC import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import random
from utils.utils import *
from data_factory.data_loader import get_loader_segment
from einops import rearrange
from metrics.metrics import *
import warnings
from augmentations import DataTransform
import torch.backends.cudnn as cudnn
warnings.filterwarnings('ignore')


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def save_checkpoint(dataset, model, path):
    torch.save(model.state_dict(), os.path.join(path, str(dataset) + '_checkpoint.pth'))
class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                               win_size=self.win_size, mode='train', dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='thre', dataset=self.dataset)

        self.build_model()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = ABC(self.channels)
        # self.model = self.model.float()

        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                # print('inputdata',input_data.shape)
                input = input_data.transpose(1, 2)
                # print('input', input.shape)
                z1,z2=DataTransform(input,self.jitter_scale_ratio,self.max_seg,self.jitter_ratio)
                z1= z1.transpose(1, 2).to(self.device)
                z2= z2.transpose(1, 2).to(self.device)
                # print('z1', z1.shape)
                # print('z2', z2.shape)
                restruct_z1,restruct_z2,attMat1,attMat2,P = self.model(z1,z2)
                # print('p', P.shape)

                resturct_loss1 = 0.0
                resturct_loss2 = 0.0
                attn_kl_loss1 = 0.0
                attn_kl_loss2 = 0.0

                resturct_loss1 += self.criterion(restruct_z1,z1)
                resturct_loss2 += self.criterion(restruct_z2,z2)
                resturct_loss=(resturct_loss1+resturct_loss2)/2
                attn_kl_loss1 += (torch.mean(my_kl_loss(attMat1, attMat2.detach())) + torch.mean(my_kl_loss(attMat2.detach(),attMat1)))
                attn_kl_loss2 += (torch.mean(my_kl_loss(attMat1.detach(), attMat2)) + torch.mean(my_kl_loss(attMat2, attMat1.detach())))
                ####
                #attn_loss=(attn_kl_loss1+attn_kl_loss2)/2
                attn_loss = (attn_kl_loss1 - attn_kl_loss2) / 2
                # attn_loss = (attn_kl_loss1) / 2
                λ=self.λ

                #正则项
                N = P.size(-1)
                # trace_loss = 1.0-torch.einsum('bii->b', P).mean() / N
                identity = torch.eye(self.win_size, device=P.device).unsqueeze(0).expand(self.batch_size, -1, -1)
                Ploss = F.mse_loss(P, identity)*N

                loss= λ*resturct_loss + attn_loss #+ 25*Ploss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s; loss: {:.4f}'.format(speed, left_time,loss))
                    #############
                    print('\tloss1: {:.4f};loss2: {:.4f}'.format(attn_kl_loss1,attn_kl_loss2))

                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.optimizer.step()

            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time))
            save_checkpoint(self.dataset, self.model, path)
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

#KLscore = my_kl_loss(attMat1.detach(), attMat2.detach()) + my_kl_loss(attMat2.detach(), attMat1.detach())
    def test(self):
        self.model.load_state_dict(
            torch.load(
                #os.path.join(str(self.model_save_path), str(self.data_path) + '_checkpoint.pth')))
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50 #乘以温度系数，防止KLscore计算softmax时分母出现过小值
        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):

            input = input_data.transpose(1, 2)
            z1,z2=DataTransform(input,self.jitter_scale_ratio,self.max_seg,self.jitter_ratio)
            z1 = z1.transpose(1, 2).to(self.device)
            z2 = z2.transpose(1, 2).to(self.device)
            ###
            restruct_z1, restruct_z2, attMat1, attMat2,P = self.model(z1, z2)

            KLscore = my_kl_loss(attMat1.detach(), attMat2.detach()) + my_kl_loss(attMat2.detach(), attMat1.detach()) * temperature
            metric = torch.softmax((KLscore), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):

            input = input_data.transpose(1, 2)
            z1,z2=DataTransform(input,self.jitter_scale_ratio,self.max_seg,self.jitter_ratio)
            z1 = z1.transpose(1, 2).to(self.device)
            z2 = z2.transpose(1, 2).to(self.device)
            restruct_z1, restruct_z2, attMat1, attMat2,P = self.model(z1, z2)

            KLscore = my_kl_loss(attMat1.detach(), attMat2.detach()) + my_kl_loss(attMat2.detach(), attMat1.detach()) * temperature

            metric = torch.softmax(KLscore, dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):

            input = input_data.transpose(1, 2)
            z1,z2=DataTransform(input,self.jitter_scale_ratio,self.max_seg,self.jitter_ratio)
            z1 = z1.transpose(1, 2).to(self.device)
            z2 = z2.transpose(1, 2).to(self.device)
            restruct_z1, restruct_z2, attMat1, attMat2,P = self.model(z1, z2)

            KLscore = my_kl_loss(attMat1.detach(), attMat2.detach()) + my_kl_loss(attMat2.detach(),attMat1.detach()) * temperature
            metric = torch.softmax(KLscore, dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        matrix = [self.index]
        # scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
        # for key, value in scores_simple.items():
        #     matrix.append(value)
        #     print('{0:21} : {1:0.4f}'.format(key, value))

        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision,
                                                                                                   recall, f_score))

        if self.data_path == 'UCR' or 'UCR_AUG':
            import csv
            with open('result/' + self.data_path + '.csv', 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(matrix)

        return accuracy, precision, recall, f_score