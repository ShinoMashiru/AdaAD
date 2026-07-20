import torch
import torch.nn as nn
import numpy as np
from attention import Attention,Transformer_encoder

class SinkhornLayer(nn.Module):
    def __init__(self, n_iters=20, temperature=0.5):
        super().__init__()
        self.n_iters = n_iters
        self.temperature = temperature

    def forward(self, scores):
        # scores: [B, N, N] logits
        log_alpha = scores / self.temperature

        # 数值稳定技巧：减去 max 防止 exp 溢出
        log_alpha = log_alpha - log_alpha.max(dim=-1, keepdim=True)[0]
        log_alpha = log_alpha - log_alpha.max(dim=-2, keepdim=True)[0]

        # log-space Sinkhorn normalization
        for _ in range(self.n_iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)

        # 转回 normal space
        P = torch.exp(log_alpha)  # still [B, N, N]
        return P

class ReorderBlock(nn.Module):
    def __init__(self, d_model, hidden_dim=64, sinkhorn_iters=20, temperature=0.1):
        super().__init__()
        # 小型前馈网络，用于编码每个输入位置的表示
        self.encoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),  # [B, N, d] → [B, N, hidden]
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)   # [B, N, hidden] → [B, N, d]
        )
        # 用于生成 pairwise scores（你可以替换为 dot-product）
        self.scorer = nn.Linear(d_model, d_model)
        # Sinkhorn模块，输出可微的近似排列矩阵
        self.sinkhorn = SinkhornLayer(sinkhorn_iters, temperature)
    def forward(self, x):
        # 输入：x 是 [B, N, d]，B是batch，N是序列长度，d是特征维度
        B, N, d = x.shape
        # 将每个位置的向量编码为更高阶的表示（可理解为位置嵌入 +上下文编码）
        x_embed = self.encoder(x)  # 仍然是 [B, N, d]
        # 计算 pairwise 打分矩阵，表示位置 i 应该去 j 的“意愿”强度
        scores = torch.bmm(x_embed, x_embed.transpose(1, 2))  # [B, N, N]
        # 应用 Sinkhorn 算子，把打分矩阵归一化为近似排列矩阵
        P = self.sinkhorn(scores)  # [B, N, N]，双随机矩阵（soft permutation）
        # 将排列矩阵作用到输入上，实现“软”重排
        x_reordered = torch.bmm(P, x)  # [B, N, d]，即 P × x
        return x_reordered, P  # 返回重排后的输入 和 近似排列矩阵



class ABC(nn.Module):
    def __init__(self, channels):
        super(ABC, self).__init__()
        self.lsoftmax = nn.LogSoftmax()
        # self.device = device
        self.Restructure = Transformer_encoder(dim=channels, depth=4,heads=4, mlp_dim=128,dropout=0)
        self.AttnMat = Attention(dim=channels, heads=4,dropout=0)

        ##newblock
        self.reorderblock = ReorderBlock(d_model=channels, sinkhorn_iters=20, temperature=0.1)

    def forward(self, features_aug1, features_aug2):
        z_aug1 = features_aug1  # features are (batch_size, #channels, seq_len)
        # z_aug1 = z_aug1.transpose(1, 2) #b n d
        z_aug2_temp = features_aug2
        # z_aug2 = z_aug2.transpose(1, 2)

        #z_aug2,P=z_aug2_temp,0 #sinkhorn-0
        z_aug2,P = self.reorderblock(z_aug2_temp) #sinkhorn-1

        restruct_z1 = self.Restructure(z_aug2, mask=None) #restruct strong to weak 1
        #restruct_z1 = z_aug2                             #restruct strong to weak 0

        restruct_z2 = self.Restructure(z_aug1, mask=None) #restruct weak to strong 1
        #restruct_z2 = z_aug1                             #restruct weak to strong 0

        #AttnMat b h n n
        attMat1 = self.AttnMat(restruct_z1,qk_flag=1, mask=None)
        attMat2 = self.AttnMat(restruct_z2,qk_flag=1, mask=None)

        return restruct_z1,restruct_z2,attMat1,attMat2,P