import torch
from torch import nn
from torch.nn.attention.flex_attention import flex_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, num_kv_heads, num_heads, hidden_size, drop = 0.1,eager = False,max_len = None):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        head_dim = hidden_size // num_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads

        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_value_proj = nn.Linear(hidden_size, num_kv_heads * head_dim * 2)

        self.out_proj = nn.Linear(num_heads * head_dim, hidden_size)
        self.scale = head_dim ** -0.5
        self.dropout = nn.Dropout(drop)
        self.eager = eager

        if eager:
            if max_len is None:
                max_len = 3000 // 4
            self.rel_pos_emb = nn.Parameter(torch.randn(2 * max_len - 1, head_dim))

    def forward(self, x:torch.Tensor):
        batch_size, seq_len, _ = x.shape

        query = self.query_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key_value = self.key_value_proj(x).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim * 2)
        keys, values = key_value.chunk(2, dim = -1 )

        query = query.permute(0, 2, 1, 3).contiguous()
        key = keys.permute(0, 2, 1, 3).contiguous()
        value = values.permute(0, 2, 1, 3).contiguous()

        if self.eager:
            output = self.eager_attention(query, key, value)
        else:
            output = self._attention(query, key, value)

        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        output = self.out_proj(output)
        return output

    def get_relative_position_bias(self, q):
        seq_len = q.shape[-2]
        positions = torch.arange(seq_len, device=q.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)

        relative_positions = torch.clamp(relative_positions, -seq_len + 1, seq_len - 1)
        relative_positions += seq_len - 1

        rel_emb = self.rel_pos_emb[relative_positions]
        rel_bias = rel_emb.mean(dim=-1)

        return rel_bias

    def eager_attention(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor):

        k = torch.repeat_interleave(k, self.num_key_value_groups, dim = 1)
        v = torch.repeat_interleave(v, self.num_key_value_groups, dim = 1)
        print(k.shape)

        scores = torch.einsum('bnqd, bnkd -> bnqk', q, k) * self.scale

        rel_bias = self.get_relative_position_bias(q)
        scores = scores + rel_bias.unsqueeze(0).unsqueeze(0)

        attn_weights = nn.functional.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        results = torch.einsum('bnqk, bnvd -> bnqd', attn_weights, v)
        return results

    def _attention(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor):
        def score_mod(score, b, h, q_idx, kv_idx):
            return score + (q_idx - kv_idx)
        results = flex_attention(q, k, v, score_mod=score_mod, enable_gqa=True, scale=self.scale)
        return results

class MHAModule(nn.Module):
    def __init__(self, num_kv_heads, num_heads, hidden_size, drop = 0.1, eager = False):
        super().__init__()
        self.attention = MultiHeadAttention(num_kv_heads, num_heads, hidden_size, drop, eager)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(drop)

    def forward(self, x:torch.Tensor):
        residual = x
        x = self.layer_norm(x)
        attn = self.attention(x)
        attn = self.dropout(attn)
        output = residual + attn
        return output


if __name__ == "__main__":
    q = torch.randn((2, 9, 10, 64))
    k = torch.randn((2, 3, 10, 64))
    v = torch.randn((2, 3, 10, 64))

    attn = MultiHeadAttention(num_kv_heads=3, num_heads=9, hidden_size=64, eager=True)
    output = attn.eager_attention(q,k,v)
    flex_output = attn._attention(q, k, v)
    print(output.shape)
    print(flex_output.shape)
    print(torch.allclose(output, flex_output, atol=1e-6))



