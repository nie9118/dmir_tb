
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim, nn, autograd
from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
import math
import torch.nn.init as init
# from RevIN import RevIN


class RevIN(nn.Module): # 插件
    """Reinversible instance normalization.

    Attributes:
      num_features: the number of features or channels
      eps: a value added for numerical stability
      axis: axis to be normalized
      affine: if True, RevIN has learnable affine parameters
    """

    def __init__(self, num_features, eps=1e-5, affine=False, axis=1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.axis = axis
        self.affine = affine
        if self.affine:
            self.init_params()

    def forward(self, x, mode):
        """Apply normalization or inverse normalization.

        Args:
          x: input ts
          mode: normalize or denormalize

        Returns:
          nomarlized or denormalized x

        Raises:
          NotImplementedError:
        """

        if mode == "norm":
            self.get_statistics(x)
            x = self.normalize(x)
        elif mode == "denorm":
            x = self.denormalize(x)
        else:
            raise NotImplementedError
        return x

    def init_params(self):
        # initialize RevIN params:
        self.affine_weight = nn.Parameter(torch.ones(1, 1,
                                                     self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1,
                                                    self.num_features))

    def get_statistics(self, x):
        self.mean = torch.mean(x, dim=self.axis, keepdim=True)
        self.stdev = torch.sqrt(torch.std(x, dim=self.axis, keepdim=True) + self.eps)

    def normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


def expand_tensor(env_emb, d):
    """Expand a tensor of shape (n, m) to (n, d, m) where every d values are the same."""
    env_emb_expanded = env_emb.unsqueeze(1)
    env_emb_repeated = env_emb_expanded.repeat(1, d, 1)
    return env_emb_repeated

class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    Paper link: https://proceedings.mlr.press/v162/zhou22g.html
    """

    def __init__(self, configs, version='fourier', mode_select='random', modes=32):

        super(Model, self).__init__()
# class Informer(nn.Module):
#     #self.S 和 self.B 是有用处的

    # def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
    #             factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
    #             dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
    #             output_attention = False, distil=True, mix=True,
    #             device=torch.device('cuda:0'),penalty_anneal_iters=5,sigma=-1.0,alpha=-1.0,scale_weight=0.1):
    #     super(Informer, self).__init__()

        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.penalty_anneal_iters=5
        # EncDoc-m6
        self.device = configs.gpu
        # EncDoc-OT1:targrt 变量的预测结果
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding(1, configs.d_model,configs.embed, configs.freq, configs.dropout)
        Attn = ProbAttention if configs.attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                                configs.d_model, configs.n_heads, mix=False),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers-1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                                configs.d_model, configs.n_heads, mix=configs.mix),
                    AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                                configs.d_model, configs.n_heads, mix=False),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )   
        self.d_model=configs.d_model
        self.c_out=configs.c_out
        self.ln=nn.Linear(configs.d_model, configs.c_out, bias=False)
        self.S = nn.Parameter(torch.empty(configs.pred_len))
        init.normal_(self.S, mean=1.0, std=1.0) 
        self.B = nn.Parameter(torch.empty(configs.pred_len))
        init.normal_(self.B, mean=0.0, std=1.0)

        self.use_RevIN  = 1
        if self.use_RevIN==1:
            self.revin = RevIN(configs.enc_in)
            self.revin_dec = RevIN(1)
        scale_weight=-1.0
        bias_weight=-1.0
        self.scale = nn.Parameter(torch.ones(1, 1,configs.enc_in-1) * scale_weight)
        self.bias = nn.Parameter(torch.ones(1, 1, configs.enc_in-1)*bias_weight)
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, flag="test", indices=1.5,
            enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_dec=x_dec[:,:,-1].unsqueeze(-1)
        if self.use_RevIN==1:
            x_enc = self.revin(x_enc, 'norm')
            x_dec = self.revin_dec(x_dec, 'norm')
        x_enc = torch.cat([x_enc[:,:,:-1] * self.scale+self.bias, x_enc[:,:,-1:]], dim=2)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out_OT = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.ln(dec_out_OT)
        if self.use_RevIN==1:
            dec_out = self.revin(dec_out, 'denorm')
            dec_out = dec_out[:,-self.pred_len:,-1].unsqueeze(-1)
        y_inv = dec_out
        # Check for NaN or Inf after the linear layer
        # if flag == "train":
        #     return y_inv
        # elif flag == "tune":
        #     return y_inv, dec_out_OT
        # else:
        return y_inv


    






















class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, additional_emb=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.device = device
        print("self.device",self.device)
        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, additional_emb, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, additional_emb, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    additional_emb, n_heads, mix=False),
                        additional_emb,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        additional_emb
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(additional_emb)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                additional_emb, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                additional_emb, n_heads, mix=False),
                    additional_emb,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(additional_emb)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=additional_emb, out_channels=c_out, kernel_size=1, bias=True)
        self.ln_add = nn.Linear(additional_emb, c_out, bias=True)
        self.ln_OT = nn.Linear(d_ff, c_out, bias=True)
        self.ln_con=nn.Linear(c_out, c_out, bias=True)
        for lin in [self.ln_add, self.ln_OT,self.ln_con]:
            nn.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]
