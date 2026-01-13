import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class patch(nn.Module):
    def __init__(self,k):
        super(patch, self).__init__()
        self.k = k
    def pad(self,x, k):
        n = x.size(-1)
        if n >= k:
            return x[..., :k]
        m = k - n
        t = (m + n - 1) // n
        repeated = x.repeat(*([1] * (x.dim() - 1) + [t]))
        padding = repeated[..., -m:]
        return torch.cat([x, padding], dim=-1)
    def forward(self, x):
        L=x.shape[1]
        x = x.permute(0, 2, 1)
        k= x.shape[-1]+self.k-1
        x = self.pad(x, k)
        x=x.permute(0, 2, 1).unsqueeze(1)
        OUT=[]
        for i in range(L):
            out = x[:, :, i:(i+self.k),:]
            OUT.append(out)
        OUT = torch.cat(OUT, dim=1)
        return OUT

class Patch(nn.Module):
    def __init__(self, k, stride=1):
        super().__init__()
        self.k = k
        self.stride = stride

    def forward(self, x):
        # x: (B, L, C) -> (B, num_patches, C, k)
        return x.unfold(1, self.k, self.stride)

class SpareEmbed3(nn.Module):
    def __init__(self, inn, seq=7, k=12, d_model=1, c_in=12):
        super(SpareEmbed3, self).__init__()
        self.k = k
        self.patch = Patch(self.k)
        self.patch_num = seq - k + 1
        self.T = nn.Conv1d(1, d_model, kernel_size=63, stride=1, padding=31, bias=False)
        self.P = nn.Linear(self.k * c_in, 32)
        self.P2 = nn.Conv1d(self.patch_num,16, kernel_size=3, stride=1, padding=1, bias=False)
        self.G = nn.GRU(32, inn, batch_first=True)
        self.se = SEBlock(self.patch_num, 4)
        self.fatten = nn.Flatten()

    def forward(self, x):

        batchSize = x.shape[0]

        tmpX = self.patch(x)[:, :self.patch_num, :, :]  # (batch, patch_num, k, c_in)
        tmpX = self.se(tmpX)
        tmpX = tmpX.reshape(batchSize, self.patch_num, -1).contiguous()  # (batch, patch_num, k*c_in)
        tmpX = self.P(tmpX)  # (batch, patch_num, 128)

        tmpX = self.P2(tmpX)  # (batch, 128, 16)

        _, tmpX = self.G(tmpX)  # (1, batch, inn)
        tmpX = tmpX.permute(1, 0, 2).contiguous()  # (batch, 1, inn)
        tmpX = self.T(tmpX)  # (batch, d_model, inn)
        return tmpX


class SpareEmbed2(nn.Module):
    def __init__(self, d_model, seq=16, k=16, conv_channels=4, embed_type='fixed', freq='h'):
        super(SpareEmbed2, self).__init__()
        self.k = k
        self.patch2 = Patch(self.k,stride=4)
        self.patch_num = (seq - self.k) // 4
        self.d_model = d_model
        self.conv_channels = conv_channels
        self.T = nn.Conv1d(1, self.conv_channels, kernel_size=4, stride=4)
        self.flatten = nn.Flatten()
        self.P1 = nn.Linear(self.k * self.conv_channels//4, 64)
        self.P = nn.Linear(64 * self.patch_num, self.d_model)

    def forward(self, x, x_mark=None):
        # 输入x shape: (batch, channel, timeLength)
        batchSize = x.shape[0]
        channel = x.shape[1]
        timeLength = x.shape[2]

        # 优化：减少permute次数
        x = x.permute(0, 2, 1).contiguous()  # (batch, timeLength, channel)
        tmpX = self.patch2(x)  # (batch, timeLength, k, channel)

        tmpX = tmpX[:, :self.patch_num, :, :]
        tmpX = tmpX.permute(0, 2, 1, 3).contiguous()
        tmpX = tmpX.view(-1, 1, self.k)  # (B*C*patch_num, 1, k)

        tmpX = self.T(tmpX)  # (batch*channel*patch_num, conv_channels, k)
        tmpX = self.flatten(tmpX)  # (batch*channel*patch_num, conv_channels*k)
        tmpX = self.P1(tmpX)  # (batch*channel*patch_num, 8)
        tmpX = tmpX.reshape(batchSize, channel, -1).contiguous()  # (batch, channel, patch_num*8)
        tmpX = self.P(tmpX)  # (batch, channel, d_model)
        hn = tmpX.reshape(batchSize, -1, self.d_model).permute(0, 2, 1).contiguous()
        return hn

class MDataEmbedding(nn.Module):
    def __init__(self, seq_in, d_model, embed_type='fixed', freq='h', dropout=0.1,k=16,t_model=None, enc_in =1,x_mark = None):
        super(MDataEmbedding, self).__init__()
        if freq == 'h': d=4
        elif freq == 't': d=5
        elif freq == 'd': d=3
        self.x_mark = x_mark
        self.spare2 = SpareEmbed2(d_model,k=16,seq=seq_in)
        if self.x_mark is None:
            self.spare3 = SpareEmbed3(enc_in,seq=seq_in,d_model=d_model,c_in=enc_in)
        else:
            self.spare3 = SpareEmbed3(enc_in+d,seq=seq_in,d_model=d_model,c_in=enc_in+d)
        self.dropout = nn.Dropout(p=dropout)
        self.weight = nn.Parameter(torch.tensor(-1.5), requires_grad=True)

    def forward(self, x, x_mark):
        alpha = torch.sigmoid(self.weight)
        x = torch.cat((x, x_mark), dim=-1)
        y =self.dropout(self.spare3(x))# (batch, d_model, inn)
        x = self.spare2(x.permute(0,2,1))
        out =  alpha * x + (1 - alpha) * y #alpha * x +
        return out,alpha * x,(1 - alpha) * y# (batch, d_model, inn)