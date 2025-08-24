import math

import torch
from torch import nn, Tensor
from src.config import NemoConformerConfig
from src.module.attention.attention import MultiHeadAttention

class NemoRelPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate, max_len, xscale, dropout_rate_emb):
        super().__init__()
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.max_len = max_len
        self.xscale = xscale
        self.dropout_rate_emb = dropout_rate_emb
        self.dropout = nn.Dropout(dropout_rate)
        if dropout_rate_emb > 0:
            self.dropout_emb = nn.Dropout(dropout_rate_emb)
        else:
            self.dropout_emb = None

    def create_pe(self, positions, dtype):
        pos_length = positions.size(0)
        pe = torch.zeros(pos_length, self.d_model, device = positions.device)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype = torch.float32, device = positions.device) * -(math.log(10000)/ self.d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0).to(dtype)
        if hasattr(self, 'pe'):
            self.pe = pe
        else:
            self.register_buffer('pe', pe, persistent=False)

    def extend_pe(self, length, device, dtype):
        needed_size = 2 * length -1
        if hasattr(self, 'pe') and self.pe.size(1) >= needed_size:
            return
        positions = torch.arange(length -1, -length, -1, dtype = torch.float32, device = device).unsqueeze(1)
        self.create_pe(positions = positions, dtype = dtype)

    def forward(self, x:torch.Tensor, cache_len = 0):
        input_len = x.size(1) + cache_len
        x = x* self.xscale
        pos_emb = self.pe[:, :input_len]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        x = x + pos_emb
        return self.dropout(x), pos_emb

class NemoLocalAttRelPositionalEncoding(NemoRelPositionalEncoding):
    def __init__(self, att_context_size, **kwargs):
        super().__init__(**kwargs)
        self.left_context, self.right_context = att_context_size[0], att_context_size[1]

    def extend_pe(self, length, device, dtype):
        if hasattr(self, 'pe'):
            return

        positions = torch.arange(
            self.left_context, -self.right_context - 1, -1, dtype=torch.float32, device=device
        ).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def forward(self, x, cache_len=0):
        if self.xscale:
            x = x * self.xscale

        end_pos = self.left_context + self.right_context + 1
        pos_emb = self.pe[:, :end_pos]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        return self.dropout(x), pos_emb

class RelPositionMultiHeadAttention(nn.Module):
    def __init__(self,
                 n_head,
                 n_feat,
                 dropout_rate,
                 pos_bias_u,
                 pos_bias_v,
                 max_cache_len=0,
                 use_bias=True,
                 use_pytorch_sdpa=False,
                 use_pytorch_sdpa_backends=None
                 ):
        super().__init__()
        self.use_pytorch_sdpa = use_pytorch_sdpa
        if self.use_pytorch_sdpa and use_pytorch_sdpa_backends:
            use_pytorch_sdpa_backends = list(
                map(
                    lambda backend_name: getattr(torch.nn.attention.SDPBackend, backend_name),
                    use_pytorch_sdpa_backends,
                )
            )
        self.use_pytorch_sdpa_backends = use_pytorch_sdpa_backends

        self.cache_drop_size = None
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_v = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.dropout = nn.Dropout(p=dropout_rate)

        self._max_cache_len = max_cache_len
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        if pos_bias_u is None or pos_bias_v is None:
            self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            nn.init.zeros_(self.pos_bias_u)
            nn.init.zeros_(self.pos_bias_v)
        else:
            self.pos_bias_u = pos_bias_u
            self.pos_bias_v = pos_bias_v

    def rel_shift(self, x):
        b, h, qlen, pos_len = x.size()  # (b, h, t1, t2)
        # need to add a column of zeros on the left side of last dimension to perform the relative shifting
        x = torch.nn.functional.pad(x, pad=(1, 0))  # (b, h, t1, t2+1)
        x = x.view(b, h, -1, qlen)  # (b, h, t2+1, t1)
        # need to drop the first row
        x = x[:, :, 1:].view(b, h, qlen, pos_len)  # (b, h, t1, t2)
        return x

    def forward(self, query, key, value, mask, pos_emb, cache=None):
        key, value, query, cache = self.update_cache(key=key, value=value, query=query, cache=cache)

        if torch.is_autocast_enabled():
            query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            q, k, v = self.forward_qkv(query, key, value)
            q = q.transpose(1, 2)  # (batch, time1, head, d_k)

            n_batch_pos = pos_emb.size(0)
            n_batch = value.size(0)
            p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
            p = p.transpose(1, 2)  # (batch, head, time1, d_k)

            # (batch, head, time1, d_k)
            q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
            # (batch, head, time1, d_k)
            q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

            matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
            matrix_bd = self.rel_shift(matrix_bd)

            if self.use_pytorch_sdpa:
                scale_factor = 1 / math.sqrt(q_with_bias_u.size(-1))
                matrix_bd = matrix_bd[:, :, :, : k.size(-2)] * scale_factor

                if mask is not None:
                    mask = mask.unsqueeze(1)
                    matrix_bd.masked_fill_(mask, -10000)

                dropout_rate = self.dropout_rate if self.training else 0
                if self.use_pytorch_sdpa_backends:
                    with torch.nn.attention.sdpa_kernel(self.use_pytorch_sdpa_backends):
                        out = torch.nn.functional.scaled_dot_product_attention(
                            q_with_bias_u, k, v, attn_mask=matrix_bd, dropout_p=dropout_rate
                        )
                else:
                    out = torch.nn.functional.scaled_dot_product_attention(
                        q_with_bias_u, k, v, attn_mask=matrix_bd, dropout_p=dropout_rate
                    )

                if mask is not None:
                    all_masked_rows = torch.all(mask, dim=-1)
                    all_masked_rows.unsqueeze_(-1)
                    all_masked_rows = all_masked_rows.expand(-1, out.size(1), -1, out.size(-1))
                    out = out.masked_fill(all_masked_rows, 0.0)

                out = out.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
                out = self.linear_out(out)  # (batch, time1, d_model)
            else:
                matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
                matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]
                scores = (matrix_ac + matrix_bd) / self.s_d_k  # (batch, head, time1, time2)
                out = self.forward_attention(v, scores, mask)

        if cache is None:
            return out
        else:
            return out, cache


