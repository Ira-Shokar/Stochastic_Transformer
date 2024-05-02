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


import torch, nn

class Stochastic_Transformer_1D(torch.nn.Module):
    def __init__(self, dim=256, seq_len=5, num_heads=16, depth=4):
        super().__init__()

        self.dim = dim

        if dim <= 32: num_heads = dim//4

        self.PE = torch.nn.Parameter(torch.zeros(seq_len, dim))

        self.att_block_in  = Attention_Block(dim, num_heads, seq_len, beta=True, stochastic=True)
        self.att_block_out = Attention_Block(dim, num_heads, seq_len, beta=True, final=True)

        self.intermediate_blocks = torch.nn.ModuleList(
            [Attention_Block(dim, num_heads, seq_len, beta=True) for _ in range(depth-2)],
            )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            #torch.nn.GELU(),
            #torch.nn.Linear(dim*4, dim)
        )

        self.register_buffer('ones_sk', \
            torch.einsum('s,k->sk', torch.ones(seq_len), torch.fft.fftfreq(dim)*dim))

    def shift_phase(self, z):
        z    = torch.fft.fft(z, norm='forward')
        phi  = torch.angle(z[:, -1, 1])
        phi  = torch.einsum('...i,sk->...isk', phi, self.ones_sk)
        z   *= torch.exp(-1j*phi)
        z    = torch.fft.ifft(z, norm='forward').real
        return z, phi[:, -1]

    def unshift_phase(self, z, phi):
        z  = torch.fft.fft(z, norm='forward')
        z *= torch.exp(1j*phi)
        z  = torch.fft.ifft(z, norm='forward').real
        return z
        
    def forward(self, z):

        beta = z[..., -1].unsqueeze(-1)
        z    = z[..., :-1]

        z, phi = self.shift_phase(z)

        z    = z + self.PE
        z, a = self.att_block_in(z, beta)
        for block in self.intermediate_blocks:
            z, _ = block(z, beta)
        z, _ = self.att_block_out(z, beta)

        z = self.mlp(z)

        z = self.unshift_phase(z, phi)

        return z, a


class Stochastic_Transformer_2D(torch.nn.Module):
    ''' Not yet completed '''
    def __init__(self):
        super().__init__()

    def forward(self):
        return 0


class Stochastic_Transformer_3D(torch.nn.Module):
    ''' Not yet completed '''
    def __init__(self):
        super().__init__()

    def forward(self):
        return 0