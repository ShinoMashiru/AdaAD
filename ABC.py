import torch
import torch.nn as nn
import numpy as np
from attention import Attention,Transformer_encoder


class ABC(nn.Module):
    def __init__(self, channels):
        super(ABC, self).__init__()
        self.lsoftmax = nn.LogSoftmax()
        # self.device = device
        self.Restructure = Transformer_encoder(dim=channels, depth=4,heads=4, mlp_dim=128,dropout=0)
        self.AttnMat = Attention(dim=channels, heads=4,dropout=0)

    def forward(self, features_aug1, features_aug2):
        z_aug1 = features_aug1  # features are (batch_size, #channels, seq_len)
        # z_aug1 = z_aug1.transpose(1, 2) #b n d
        z_aug2 = features_aug2
        # z_aug2 = z_aug2.transpose(1, 2)

        restruct_z1 = self.Restructure(z_aug2, mask=None)
        restruct_z2 = self.Restructure(z_aug1, mask=None)

        #AttnMat b h n n
        attMat1 = self.AttnMat(restruct_z1,qk_flag=1, mask=None)
        attMat2 = self.AttnMat(restruct_z2,qk_flag=1, mask=None)

        return restruct_z1,restruct_z2,attMat1,attMat2