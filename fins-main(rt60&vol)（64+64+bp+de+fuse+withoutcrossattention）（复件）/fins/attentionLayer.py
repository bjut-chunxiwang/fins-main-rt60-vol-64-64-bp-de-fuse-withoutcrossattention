import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention


class attentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(attentionLayer, self).__init__()
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, tar):
        # Assume src and tar have shape (batch_size, d_model)

        # Step 1: Add an extra dimension to src and tar to make them (1, batch_size, d_model)
        src = src.unsqueeze(0)  # Shape becomes (1, batch_size, d_model)
        tar = tar.unsqueeze(0)  # Shape becomes (1, batch_size, d_model)

        # Step 2: Multihead Attention
        # src2 = self.self_attn(tar, src, src, attn_mask=None, key_padding_mask=None)[0]

        # Step 3: Add residual connection and apply layer normalization
        src = src + self.dropout1(src)
        src = self.norm1(src)

        # Step 4: Feedforward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # Step 5: Remove the extra dimension to return to (batch_size, d_model)
        src = src.squeeze(0)  # Shape becomes (batch_size, d_model)

        return src
