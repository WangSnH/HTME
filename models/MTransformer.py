import torch
import torch.nn as nn

from layers.Transformer_EncDec import Encoder,EncoderLayer

from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import MDataEmbedding
from layers.RevIN import RevIN
import numpy as np


class Model(nn.Module):


    def __init__(self, configs):
        super(Model, self).__init__()
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention


        self.enc_in = configs.enc_in

        self.enc_embedding = MDataEmbedding(self.seq_len, configs.d_model, configs.embed, configs.freq,configs.dropout,k=16,t_model=configs.t_model,enc_in=self.enc_in)
        self.mean_embedding = MDataEmbedding(self.seq_len, configs.d_model, configs.embed, configs.freq,configs.dropout,k=16,t_model=configs.t_model,enc_in=self.enc_in)
        self.mencoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.t_model, configs.n_heads),
                    configs.t_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.t_model)
        )

        self.projector1 = nn.Linear(configs.t_model, configs.pred_len, bias=True)
        self.projector2 = nn.Linear(configs.t_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc  = self.revin_layer(x_enc,'norm')
        enc_out = self.enc_embedding(x_enc, x_mark_enc).permute(0,2,1)

        enc_out, attns = self.mencoder(enc_out, attn_mask=None)
        enc_out = self.projector1(enc_out)

        dec_out = enc_out.permute(0,2,1)[:,:,:x_dec.shape[-1]]

        dec_out = self.revin_layer(dec_out, 'denorm')
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]