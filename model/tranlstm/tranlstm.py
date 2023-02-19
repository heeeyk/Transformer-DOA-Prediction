import torch
import math
from torch.nn.modules import Transformer, TransformerEncoder, TransformerEncoderLayer, LayerNorm
import torch.nn as nn
import torch.nn.functional as F
from model.base_transformer.transformer_xl.masks import generate_square_subsequent_mask
from torch import Tensor
from typing import Dict, Tuple, Optional, List
import copy
import imp


class MyTransformer(torch.nn.Module):
    def __init__(
            self,
            device,
            number_time_series: int,
            seq_length: int = 48,
            output_seq_len: int = None,
            d_model: int = 128,
            n_heads: int = 8,
            dropout=0.1,
            forward_dim=2048,
            sigmoid=False):
        """A full transformer model
        :param number_time_series: The total number of time series present
            (e.g. n_feature_time_series + n_targets)
        :type number_time_series: int
        :param seq_length: The length of your input sequence, defaults  to 48
        :type seq_length: int, optional
        :param output_seq_len: The length of your output sequence, defaults
            to None
        :type output_seq_len: int, optional
        :param d_model: The dimensions of your model, defaults to 128
        :type d_model: int, optional
        :param n_heads: The number of heads in each encoder/decoder block,
            defaults to 8
        :type n_heads: int, optional
        :param dropout: The fraction of dropout you wish to apply during
            training, defaults to 0.1 (currently not functional)
        :type dropout: float, optional
        :param forward_dim: Currently not functional, defaults to 2048
        :type forward_dim: int, optional
        :param sigmoid: Whether to apply a sigmoid activation to the final
            layer (useful for binary classification), defaults to False
        :type sigmoid: bool, optional
        """
        super().__init__()
        if output_seq_len is None:
            output_seq_len = seq_length
        self.out_seq_len = output_seq_len
        self.mask = generate_square_subsequent_mask(seq_length).to(device)
        self.dense_shape = torch.nn.Linear(number_time_series, d_model)
        self.dense_t = torch.nn.Linear(1, d_model)
        self.pe = SimplePositionalEncoding(d_model)
        self.transformer = Transformer(
            d_model,
            nhead=n_heads,
            num_encoder_layers=1,
            num_decoder_layers=1,
            device=device,
            activation=torch.nn.functional.selu,
            dropout=0)

        self.final_layer = nn.Linear(d_model, 1)
        self.permute_layer = nn.Linear(seq_length, 1)
        self.sequence_size = seq_length
        self.tgt_mask = generate_square_subsequent_mask(seq_length).to(device)

    # def forward(self, x: torch.Tensor, t: torch.Tensor, tgt_mask=None):
    #     x = self.encode_sequence(x, src_mask=self.mask)
    #     return self.decode_seq(x, t, tgt_mask)

    def forward(self, x, t, tgt_mask=None):
        x = self.encode_sequence(x, src_mask=self.mask)
        x = self.decode_seq(x, t, tgt_mask)
        x = x.permute(2, 1, 0)
        x = self.permute_layer(x)
        return x

    def basic_feature(self, x: torch.Tensor):
        x = self.dense_shape(x)
        x = self.pe(x)
        x = x.permute(1, 0, 2)
        return x

    def basic_t(self, x: torch.Tensor):
        x = self.dense_t(x)
        x = self.pe(x)
        x = x.permute(1, 0, 2)
        return x

    def encode_sequence(self, x, src_mask=None):
        # 1.维度扩展+位置编码
        x = self.basic_feature(x)
        x = self.transformer.encoder(src=x, mask=src_mask)
        return x

    def decode_seq(self, mem, t, tgt_mask=None, view_number=None) -> torch.Tensor:
        if view_number is None:
            view_number = self.out_seq_len
        if tgt_mask is None:
            tgt_mask = self.tgt_mask
        t = self.basic_t(t)
        x = self.transformer.decoder(t, mem, tgt_mask=tgt_mask)
        x = self.final_layer(x)
        return x


class GLU(nn.Module):
    """
    ----------
    input_dim: int
        The embedding size of the input.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super(GLU, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim*2)

    def forward(self, x):
        """
        :param x:(b, d) or (l, b, d)
        where l is the length of medicine history
            , b is batch size
            , d is feature dimension
        :return:x:(l, b, d)
        """
        x = F.glu(self.fc(x))
        return x


class GRN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: Optional[float] = 0.05,
                 context_dim: Optional[int] = None):
        """
        :param input_dim:The embedding width/dimension of the input.
        :param hidden_dim:The intermediate embedding width.
        :param output_dim:The embedding width of the output tensors.
        :param dropout:The dropout rate associated with the component.
        :param context_dim:The embedding width of the context signal
            expected to be fed as an auxiliary input to this component.
        """
        super().__init__()
        self.layernorm = nn.LayerNorm(output_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, hidden_dim)
        self.skip = nn.Linear(input_dim, output_dim)
        self.forw = nn.Sequential(
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
            GLU(input_dim=output_dim, output_dim=output_dim))

    def forward(self, x, c: Optional[torch.Tensor] = None):
        """
        :param x:(b, d) or (l, b, d)
        :param c:(b, c) or (l, b, c)
        :return:x:(b, o) or (l, b, o)
        where l is the length of medicine history, b is batch size ,and
            d is feature dimension, c is context dimension, o is dimension of output
        """
        res = self.skip(x)
        x = self.fc1(x)
        if c is not None:
            c = self.context_fc(c)
            x += c
        x = self.forw(x) + res
        x = self.layernorm(x)
        return x


class VariableSelectionNetwork(nn.Module):
    """
    Parameters
    ----------
    input_dim: int
        The attribute/embedding dimension of the input, associated with the ``state_size`` of th model.
    input_num: int
        The quantity of input variables, including both numeric and categorical inputs for the relevant channel.
    hidden_dim: int
        The embedding width of the output.
    dropout: float
        The dropout rate associated with ``GatedResidualNetwork`` objects composing this object.
    context_dim: Optional[int]
        The embedding width of the context signal expected to be fed as an auxiliary input to this component.
    """

    def __init__(self,
                 input_dim: int,
                 input_num: int,
                 hidden_dim: int,
                 dropout: float,
                 context_dim: Optional[int] = None):
        super(VariableSelectionNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.input_num = input_num
        self.dropout = dropout
        self.context_dim = context_dim

        self.flattened_grn = GRN(
            input_dim=self.input_num * self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.input_num,
            dropout=self.dropout,
            context_dim=self.context_dim)

        self.softmax = nn.Softmax(dim=1)

        self.single_variable_grns = nn.ModuleList()
        for _ in range(self.input_num):
            self.single_variable_grns.append(
                GRN(input_dim=self.input_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=self.hidden_dim,
                    dropout=self.dropout,
                    context_dim=context_dim))

    def forward(self, flattened_embedding, context=None):
        """
        :param flattened_embedding: (l, b, d)
        :param context: (l, b, d)
        :return:outputs:()
        l = num_input
        b = num_samples * num_temporal_steps
        d = input_dim
        """
        # x = flattened_embedding.reshape(flattened_embedding.shape[1], -1)
        x = flattened_embedding.reshape(flattened_embedding.shape[0], -1)
        # x = flattened_embedding
        # x:(b, 3)
        sparse_weights = self.flattened_grn(x, context)
        # sparse_weights:(b, h), h:hidden size
        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)
        # sparse_weights:(b, h, 1)

        processed_inputs = []
        for i in range(self.input_num):
            processed_inputs.append(
                self.single_variable_grns[i](x[..., (i * self.input_dim): (i + 1) * self.input_dim], context))
        # processed_inputs[i]:(b, h), i=1...3

        processed_inputs = torch.stack(processed_inputs, dim=-1)
        # processed_inputs:(b, h, 3)
        outputs = processed_inputs * sparse_weights.transpose(1, 2)
        # outputs:(b, h, 3)
        # outputs = outputs.sum(axis=-1)

        return outputs, sparse_weights



class StaticCovariateEncoder(nn.Module):
    def __init__(self, f_dim, hidden_dim):
        super(StaticCovariateEncoder, self).__init__()
        self.f_dim = f_dim
        self.vsn = VariableSelectionNetwork(
            # input_dim=1,
            input_dim=hidden_dim,
            input_num=4,
            hidden_dim=hidden_dim,
            dropout=0.1,
            context_dim=hidden_dim,
        )
        self.grn = nn.ModuleList([GRN(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,) for _ in range(4)])

    def forward(self, x):
        variable_ctx, sparse_weights = self.vsn(x)
        # variable_ctx:(bs, h, 4)
        variable_ctx = variable_ctx.sum(axis=-1)
        # variable_ctx:(bs, h)
        cs, ce, cc, ch = tuple(m(variable_ctx) for m in self.grn)
        # cs:(bs, 4, h)
        return cs, ce, cc, ch


class RnnEecoder(nn.Module):
    def __init__(self, f, n):
        """
        :param f: feature nums of input
        :param n: the middle layer feature nums of lstm
        """
        super().__init__()
        self.f = f
        self.n = n
        # self.lstm1 = nn.LSTM(1, self.n, batch_first=True)
        # self.lstm2 = nn.LSTM(1, self.n, batch_first=True)
        self.lstm = nn.LSTM(1, self.n, batch_first=True)
        self.gate = GLU(self.n, self.n)
        self.fc = nn.Linear(f, n)
        # self.feature_cat = nn.Linear(2*n, n)
        self.layernorm = nn.LayerNorm(n, eps=1e-5)

    def forward(self, x: torch.Tensor, c_h: torch.Tensor, c_c: torch.Tensor):
        """
        :param x: input of lstm encoder, x:(l, b, d)
        :param c_h: information of body, c_h:(b, d)
        :param c_c: information of body, c_c:(b, d)
        :return: x:(l, b, n)
        where b is batch size ,and
            d is feature dimension
        """
        res = x
        if self.f != self.n:
            res = self.fc(res)
        h0 = (c_h.unsqueeze(0), c_c.unsqueeze(0))
        x, (hn, cn) = self.lstm(x, hx=h0)
        # x1, (hn, cn) = self.lstm1(x[..., 0].unsqueeze(-1), hx=h0)
        # x2, (hn, cn) = self.lstm2(x[..., 1].unsqueeze(-1), hx=h0)
        # x = torch.cat((x1, x2), dim=2)
        # x = self.feature_cat(x)
        x = self.layernorm(self.gate(x) + res)
        return x


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        assert config.hidden_dim % config.n_head == 0
        self.i = 0
        self.d_head = config.hidden_dim // config.n_head
        self.qkv_linears = nn.Linear(config.hidden_dim, (2 * self.n_head + 1) * self.d_head, bias=True)
        self.out_proj = nn.Linear(self.d_head, config.hidden_dim, bias=True)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.out_dropout = nn.Dropout(config.dropout)
        self.scale = self.d_head**-0.5
        self.register_buffer("_mask", torch.triu(torch.full((config.example_length, config.example_length), float('-inf')), 1).unsqueeze(0))

    def forward(self, x: torch.Tensor, mask_future_timesteps: bool = True) -> Tuple[Tensor, Tensor]:
        bs, t, h_size = x.shape
        qkv = self.qkv_linears(x)
        q, k, v = qkv.split((self.n_head * self.d_head, self.n_head * self.d_head, self.d_head), dim=-1)
        q = q.view(bs, t, self.n_head, self.d_head)
        k = k.view(bs, t, self.n_head, self.d_head)
        v = v.view(bs, t, self.d_head)

        # attn_score = torch.einsum('bind,bjnd->bnij', q, k)
        attn_score = torch.matmul(q.permute((0, 2, 1, 3)), k.permute((0, 2, 3, 1)))
        attn_score.mul_(self.scale)

        if mask_future_timesteps:
            attn_score = attn_score + self._mask

        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.attn_dropout(attn_prob)

        # attn_vec = torch.einsum('bnij,bjd->bnid', attn_prob, v)
        attn_vec = torch.matmul(attn_prob, v.unsqueeze(1))
        m_attn_vec = torch.mean(attn_vec, dim=1)
        out = self.out_proj(m_attn_vec)
        out = self.out_dropout(out)

        # import matplotlib.pyplot as plt
        # if self.i == 0:
        #     print(attn_prob.shape)
        # plt.plot(torch.mean(torch.mean(attn_prob, dim=1), dim=-2).detach().cpu()[10, :])
        # plt.savefig(f'/home/user02/HYK/bis_transformer/output/tranlstm/picture/t{self.i}.png')
        # plt.close()
        # self.i += 1
        return out, attn_vec


class TemporalFusionDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, context_dim, output_dim, config):
        super().__init__()
        self.example_length = config.example_length
        self.enrichment_grn = GRN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=0.05,
            context_dim=context_dim
        )
        self.attention = InterpretableMultiHeadAttention(config)
        self.att_gate = GLU(config.hidden_dim, config.hidden_dim)
        self.attention_ln = LayerNorm(config.hidden_dim, eps=1e-3)

        self.positionwise_grn = GRN(config.hidden_dim,
                                    config.hidden_dim,
                                    dropout=config.dropout,
                                    output_dim=config.hidden_dim)
        self.decoder_gate = GLU(config.hidden_dim, config.hidden_dim)
        self.decoder_ln = LayerNorm(config.hidden_dim, eps=1e-3)
        # self.quantile = nn.Linear(config.hidden_dim, config.quantiles)
        # self.fc = nn.Sequential(
        #     nn.Linear(config.hidden_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1),
        #     # nn.Dropout()
        # )

    def forward(self, x, ce):
        res = x[:, -1, :]
        ce = ce.unsqueeze(1).repeat(1, self.example_length, 1)
        enriched = self.enrichment_grn(x, c=ce)
        x, _ = self.attention(enriched)
        x = x[:, -1, :]
        enriched = enriched[:, -1, :]
        x = self.att_gate(x)
        x = x + enriched
        x = self.attention_ln(x)
        x = self.positionwise_grn(x)
        x = self.decoder_gate(x)
        x = x + res
        x = self.decoder_ln(x)
        # x = self.quantile(x)
        # x = self.fc(x)
        return x


class TemporalFusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.static_encoder1 = VariableSelectionNetwork(
                input_dim=config.static_dim,
                input_num=config.static_num,
                hidden_dim=config.hidden_dim,
                dropout=0.1)
        self.static_encoder2 = StaticCovariateEncoder(config.hidden_dim, config.hidden_dim)

        self.varialbe_selection = VariableSelectionNetwork(
            input_dim=config.temporal_dim,
            input_num=config.example_length,
            hidden_dim=config.example_length,
            dropout=0.1,
            context_dim=config.hidden_dim,
        )
        self.rnn_encoder = RnnEecoder(1, config.hidden_dim)
        self.time_decoder = TemporalFusionDecoder(config.hidden_dim, config.hidden_dim, config.hidden_dim, config)

    def forward(self, x, b):
        b, _ = self.static_encoder1(b)
        cs, ce, ch, cc = self.static_encoder2(b)
        x, _ = self.varialbe_selection(x, cs)
        x = x.unsqueeze(-1)
        x = self.rnn_encoder(x, ch, cc)
        x = self.time_decoder(x, ce)
        return x


class LstmVsn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = 3
        self.memory_cell = 8
        self.body_dim = 4
        self.n = 16

        config.hidden_dim = self.input_dim*self.memory_cell

        self.lstm1 = nn.LSTM(1, 8, batch_first=True)
        self.lstm2 = nn.LSTM(1, 8, batch_first=True)
        self.lstm3 = nn.LSTM(1, 8, batch_first=True)
        self.fusion_lstm = nn.LSTM(24, 24, batch_first=True)


        self.static_encoder1 = VariableSelectionNetwork(
                input_dim=config.static_dim,
                input_num=config.static_num,
                hidden_dim=config.hidden_dim,
                dropout=0.1)
        self.static_encoder2 = StaticCovariateEncoder(config.hidden_dim, config.hidden_dim)

        self.time_decoder = TemporalFusionDecoder(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            context_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            config=config)

        self.fc1 = nn.Linear(8, 24)
        self.fc2 = nn.Linear(8, 24)
        self.fc3 = nn.Linear(8, 24)

        self.bottle_neck = nn.Sequential(
            nn.Linear(config.hidden_dim*4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, b):
        # x:(bs, 180, 3)
        b, _ = self.static_encoder1(b)
        # b:
        cs, ce, ch, cc = self.static_encoder2(b)
        # cs:(bs, h)
        # ce = self.bodyencoder(b)

        """
            encoder:
        """

        x1, (hn, cn) = self.lstm1(x[..., 0].unsqueeze(-1))
        x2, (hn, cn) = self.lstm2(x[..., 1].unsqueeze(-1))
        x3, (hn, cn) = self.lstm3(x[..., 2].unsqueeze(-1))
        # x1:(bs, 180, 8)
        x = torch.cat((x1, x2, x3), dim=-1)
        x, (hn, cn) = self.fusion_lstm(x)
        # x:(bs, 180, 24)
        """
            decoder:
        """
        x = self.time_decoder(x, cs)
        # x:(bs, 1, 24)
        x1 = self.fc1(x1[:, -1])
        x2 = self.fc2(x2[:, -1])
        x3 = self.fc3(x3[:, -1])
        x = torch.cat((x, x1, x2, x3), dim=-1)
        x = self.bottle_neck(x)
        return x


class LstmVsn2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = 3
        self.memory_cell = 8
        self.body_dim = 4
        self.n = 16
        self.lstm1 = nn.LSTM(1, self.memory_cell, batch_first=True)
        self.lstm2 = nn.LSTM(1, self.memory_cell, batch_first=True)
        self.lstm3 = nn.LSTM(1, self.memory_cell, batch_first=True)

        config.hidden_dim = self.input_dim*self.memory_cell
        # self.bodyencoder = GRN(
        #     input_dim=self.body_dim,
        #     hidden_dim=config.hidden_dim,
        #     output_dim=config.hidden_dim
        # )

        self.static_encoder1 = VariableSelectionNetwork(
                input_dim=config.static_dim,
                input_num=config.static_num,
                hidden_dim=config.hidden_dim,
                dropout=0.1)
        self.static_encoder2 = StaticCovariateEncoder(config.hidden_dim, config.hidden_dim)

        self.time_decoder = TemporalFusionDecoder(config.hidden_dim, config.hidden_dim, config.hidden_dim, config)

        self.fc1 = nn.Linear(8, 1)
        self.fc2 = nn.Linear(8, 1)
        self.fc3 = nn.Linear(8, 1)
        self.skip_fc = nn.Linear(self.input_dim+1, 1)

    def forward(self, x, b):
        b, _ = self.static_encoder1(b)
        cs, ce, ch, cc = self.static_encoder2(b)
        # ce = self.bodyencoder(b)

        x1, (hn, cn) = self.lstm1(x[..., 0].unsqueeze(-1))
        x2, (hn, cn) = self.lstm2(x[..., 1].unsqueeze(-1))
        x3, (hn, cn) = self.lstm3(x[..., 2].unsqueeze(-1))

        x = torch.cat((x1, x2, x3), dim=2)
        x = self.time_decoder(x, ce)
        x1 = self.fc1(x1[:, -1])
        x2 = self.fc2(x2[:, -1])
        x3 = self.fc3(x3[:, -1])
        x = torch.cat((x, x1, x2, x3), dim=1)
        x = self.skip_fc(x)
        return x


class SimplePositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(SimplePositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Creates a basic positional encoding"""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    import params_save
    args = params_save.Params.tft()
    # v = VariableSelectionNetwork(
    #     input_dim=2,
    #     input_num=180,
    #     hidden_dim=180,
    #     dropout=0.1,
    #     context_dim=32,)

    # v_x = torch.ones((100, 180, 2))
    # v_c = torch.ones((100, 32))
    # v_o = v(v_x, v_c)
    # v_o1 = v_o[0].unsqueeze(-1)
    # # v_o1:(100, 180, 1)
    #
    # r = RnnEecoder(1, 32)
    # r_x = torch.ones((100, 180, 1))
    # r_c = torch.ones((100, 32))
    # r_o = r(v_o1, r_c, r_c)
    # # r_o:(100, 180, 32)
    #
    # t = TemporalFusionDecoder(32, 64, 32, args)
    # t.apply(weights_init)
    # t_o = t(r_o, r_c)
    # # t_o:(100, 3)

    m = TemporalFusionTransformer(config=args)
    a = torch.ones((100, 30, 2))
    b = torch.ones((100, 4))
    x = m(a, b)
    # x:(100, 3)

    imp.reload(trainer)
    del box
    box = trainer.Trainer(args)
    box.train(
        X=train_loader,
        X2=vaild_loader,
        model_file=args.best_file,
        best_loss=args.best_loss,
        config=args,
        p=eff_label_dist
    )




