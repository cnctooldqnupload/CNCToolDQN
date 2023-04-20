import torch
import torch.nn as nn
import torch.nn.functional as F

from attn1 import AnomalyAttention, AttentionLayer
from embed import DataEmbedding, TokenEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, device, attention, d_model, d_ff=None, dropout=0.4, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)#.to(device)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)#.to(device)
        self.norm1 = nn.LayerNorm(d_model)#.to(device)
        self.norm2 = nn.LayerNorm(d_model)#.to(device)
        self.dropout = nn.Dropout(dropout)#.to(device)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.device = device

    def forward(self, x, attn_mask=None):
        new_x  = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
                
        x = x + self.dropout(new_x)#.to(device)
        y = x = self.norm1(x)#.to(device)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))#.to(device)
        y = self.dropout(self.conv2(y).transpose(-1, 1))#.to(device)

        return self.norm2(x + y).to(self.device) #, attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        #series_list = []
        #prior_list = []
        #sigma_list = []
                
        for attn_layer in self.attn_layers:
            
            x = attn_layer(x, attn_mask=attn_mask)
            
            #series_list.append(series)
            #prior_list.append(prior)
            #sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)
        
        return x #,sigma_list
        #return x, series_list, prior_list, sigma_list

class Attention_layer(nn.Module):
    def __init__(self, device, win_size, enc_in, c_out,  d_model= 64, n_heads= 6, e_layers=1, d_ff= 64,
                 dropout=0.2, activation='gelu', output_attention=False, ):
        super(Attention_layer, self).__init__()
        self.output_attention = output_attention

        self.device = device

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout) #.to(device)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    self.device,
                    AttentionLayer(
                        self.device,
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation = activation
                )  for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        ) 

        self.projection = nn.Linear(d_model, c_out, bias=True)

        

    def forward(self, x):
        
        x = x.to(self.device)
        
        enc_out = self.embedding(x)        
        enc_out = self.encoder(enc_out)
        enc_out = self.projection(enc_out).to(self.device)

        if self.output_attention:
            return enc_out #, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]
