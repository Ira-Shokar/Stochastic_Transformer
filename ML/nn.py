# coding=utf-8
# Copyright (c) 2024 Ira Shokar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from einops.layers.torch import Rearrange

class Multihead_Attention(torch.nn.Module):

    def __init__(self, dim: int, num_heads: int = 16, seq_len: int = 1, temporal: bool = True, stochastic: bool = False, beta: bool = False) -> None:
        super().__init__()

        self.beta       = beta
        self.stochastic = stochastic

        self.inv_sqrt_head_dim = (dim / num_heads) ** -0.5

        dim_in      = dim+1     if beta       else dim
        seq_len_out = seq_len+1 if stochastic else seq_len

        self.head_reshape = Rearrange('... p (h d) -> ... h p d', h=num_heads)

        self.q_proj = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim),
            self.head_reshape
        )

        self.kv_proj = torch.nn.Sequential(
            torch.nn.Linear(dim_in, 2*dim),
            self.head_reshape
        )

        if self.stochastic:
            self.kv_e_proj = torch.nn.Sequential(
                torch.nn.Linear(dim_in, 2*dim),
                self.head_reshape
            )

        mask = torch.triu(-float('inf') * torch.ones((seq_len, seq_len_out)), diagonal=1) if temporal else None

        self.register_buffer('mask', mask)

    def proj_stochastic(self, x:torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor = None) -> tuple:
        e = torch.rand_like(x)[..., :, :-1]

        if self.beta:
            e = torch.cat([beta, e], dim=-1)

        e = e[..., 0, :].unsqueeze(-2)

        k_e, v_e = torch.chunk(self.kv_e_proj(e), 2, dim=-1)

        k = torch.cat([k_e, k], dim=-2)
        v = torch.cat([v_e, v], dim=-2)

        return k, v

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        a = torch.einsum('...hpd,...hqd->...hpq', q, k) * self.inv_sqrt_head_dim
        if mask is not None:
            a += mask
        a = a.softmax(-1)
        o = torch.einsum('...hpq,...hqd->...phd', a, v).flatten(-2)
        return o, a

    def forward(self, x: torch.Tensor, beta: torch.Tensor = None) -> tuple:

        if self.beta:
            x = torch.cat([beta, x], dim=-1)

        q    = self.q_proj(x)
        k, v = torch.chunk(self.kv_proj(x), 2, dim=-1)

        if self.stochastic:
            k, v = self.proj_stochastic(x, k, v, beta)

        o, a = self.attention(q, k, v, mask=self.mask)

        return o, a.mean(1)


class Attention_Block(torch.nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, seq_len: int, stochastic: bool = False, beta: bool = False, final: bool = False, ) -> None:
        super().__init__()

        self.final = final

        self.ln_0 = torch.nn.LayerNorm(emb_dim)
        self.ln_1 = torch.nn.LayerNorm(emb_dim)

        self.mha = Multihead_Attention(emb_dim, num_heads, seq_len, stochastic=stochastic, beta=beta)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 4*emb_dim),
            torch.nn.GELU(),
            torch.nn.Linear(4*emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor, beta:torch.Tensor=None) -> tuple:

        o, a = self.mha(self.ln_0(x), beta)
        x    = x + o

        if self.final: x = x[:, -1]

        o = self.mlp(self.ln_1(x))
        x = x + o

        return x, a