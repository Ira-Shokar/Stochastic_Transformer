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


import torch, numpy as np, matplotlib.pyplot as plt

def plot_pdf(H_t_arr, H_p_arr, edges_arr, show=True):

    label = ['U', r'$\partial U/\partial y$', r'$\partial U/\partial t$']

    #if show==True: fig, axs = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    if show==True: fig, axs = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    else         : fig, axs = plt.subplots(4, 4, figsize=(18, 4*4) , constrained_layout=True)

    beta = [0.3, 0.6, 0.9, 1.8]

    for i, (H_t, H_p, edges) in enumerate(zip(H_t_arr, H_p_arr, edges_arr)):

        ax = axs[i, 0]
        c = ax.pcolor(edges[0].detach().cpu().numpy(),
            edges[1].detach().cpu().numpy(),
            H_t.sum(2).detach().cpu().numpy(),
            norm=LogNorm(), cmap='inferno')
        ax.set_xlabel(label[0]); ax.set_ylabel(label[1])
        ax.set_title(f'Numerical Integration PDF({label[0]}, {label[1]}), beta={beta[i]}')
        try: fig.colorbar(c, ax=ax)
        except: pass

        ax = axs[i, 1]
        c = ax.pcolor(edges[0].detach().cpu().numpy(),
            edges[1].detach().cpu().numpy(),
            H_p.sum(2).detach().cpu().numpy(),
            norm=LogNorm(), cmap='inferno')
        ax.set_xlabel(label[0]); ax.set_ylabel(label[1])
        ax.set_title(f'SLT PDF({label[0]}, {label[1]}), beta={beta[i]}')
        try: fig.colorbar(c, ax=ax)
        except: pass

        ax = axs[i, 2]
        c = ax.pcolor(edges[0].detach().cpu().numpy(),
            edges[2].detach().cpu().numpy(),
            H_t.sum(1).detach().cpu().numpy(),
            norm=LogNorm(), cmap='inferno') 
        ax.set_xlabel(label[0]); ax.set_ylabel(label[2])
        ax.set_title(f'Numerical Integration PDF({label[0]}, {label[2]}), beta={beta[i]}')
        try: fig.colorbar(c, ax=ax)
        except: pass

        ax = axs[i, 3]
        c = ax.pcolor(edges[0].detach().cpu().numpy(),
            edges[2].detach().cpu().numpy(),
            H_p.sum(1).detach().cpu().numpy(),
            norm=LogNorm(), cmap='inferno')
        ax.set_xlabel(label[0]); ax.set_ylabel(label[2])
        ax.set_title(f'SLT PDF({label[0]}, {label[2]}), beta={beta[i]}')
        try: fig.colorbar(c, ax=ax)
        except: pass

    plt.show()

def plot_1d_pdf(H_t_arr, H_p_arr, edges_arr,  show=True):

    label = [r'$U$', r'$\partial_y U$', r'$\partial_t U$']

    fig, axs = plt.subplots(4, 3, figsize=(16, 12) , constrained_layout=True)

    index = [(1, 2), (0, 2), (0, 1)]
    lims  = [1e-2, 1e-3, 1e-3, 2e-1, 2e-1, 5e-1]

    beta = [0.3, 0.6, 0.9, 1.8]

    for j, (H_t, H_p, edges) in enumerate(zip(H_t_arr, H_p_arr, edges_arr)):

        for i in range(3):
            
            h_t = H_t.sum(index[i]).detach().cpu().numpy()
            h_p = H_p.sum(index[i]).detach().cpu().numpy()
            x = np.linspace(edges[i].min(), edges[i].max(), len(h_t))

            ax = axs[j, i]

            ax.plot(x, h_t, 'g', label='Truth')
            ax.plot(x, h_p, 'r', label='ML')
            ax.set_xlabel(label[i]); ax.set_ylabel('Density')
            ax.set_title(f'Truth PDF({label[i]}), beta={beta[j]}')
            ax.legend()

    plt.show()

def plot_spectra(truth_arr, preds_arr, show=True, beta=None):

    fig, axs = plt.subplots(2, 2, figsize=(20, 10), constrained_layout=True)

    for i, (truth, preds) in enumerate(zip(truth_arr, preds_arr)):

        truth = torch.Tensor(truth).unsqueeze(0)
        preds = preds[0].unsqueeze(0)

        psd_t = torch.abs(torch.fft.rfft(truth, norm='ortho')).pow(2).flatten(0,-2)
        psd_f = torch.abs(torch.fft.rfft(preds, norm='ortho')).pow(2).flatten(0,-2)

        mean_t = psd_t.mean(0).detach().cpu().numpy()
        mean_f = psd_f.mean(0).detach().cpu().numpy()

        ax = axs[i//2, i%2]
        ax.plot(mean_t, label='Truth', c='green')
        ax.plot(mean_f, label='ML'   , c='red')
        ax.set_ylim(1e-9, 1)
        ax.set_xlim(1, 100)
        ax.set_yscale('log')
        ax.set_xscale('log', base=2)
        ax.set_xlabel('k')
        ax.set_ylabel('ε(k)')
        ax.set_title(f'Kinetic Energy Spectrum, time averaged, beta={beta[i]}')
        ax.legend()

    plt.show()



def plot_ens(truth, u, seq_len, show=True):

    fig, axs = plt.subplots(5, 4, figsize=(12*2, 6*2), constrained_layout=True)

    x = np.linspace(0, 2360, 64)
    y = np.linspace(-90, 90, 32)
    X, Y = np.meshgrid(x, y)

    for j, (truth_, u_) in enumerate(zip(truth, u)):
        imp =  axs[0, j].imshow(truth_.T, aspect='auto')
        axs[0, j].axvline(x=seq_len, ymin=0, ymax=256, c='w', linestyle='--')
        for i in range(0, 4):
            axs[i+1, j].imshow(u_[i].detach().cpu().numpy().T, aspect='auto')
            axs[i+1, j].axvline(x=seq_len, ymin=0, ymax=256, c='w', linestyle='--')

    axs[0, 0].set_ylabel("Truth (y)")
    axs[1, 0].set_ylabel("SLT (y)")
    axs[2, 0].set_ylabel("SLT (y)")
    axs[3, 0].set_ylabel("SLT (y)")
    axs[4, 0].set_ylabel("SLT (y)")

    axs[-1, 0].set_xlabel("t")
    axs[-1, 1].set_xlabel("t")
    axs[-1, 2].set_xlabel("t")
    axs[-1, 3].set_xlabel("t")

    axs[0, 0].set_title(r'$ \beta=0.3$')
    axs[0, 1].set_title(r'$ \beta=0.6$')
    axs[0, 2].set_title(r'$ \beta=0.9$')
    axs[0, 3].set_title(r'$ \beta=1.8$')


def plot_ens_int(truth, u, seq_len, show=True):

    fig, axs = plt.subplots(6, 4, figsize=(12*2, 6*2), constrained_layout=True)

    x = np.linspace(0, 2360, 64)
    y = np.linspace(-90, 90, 32)
    X, Y = np.meshgrid(x, y)

    for j, (truth_, u_) in enumerate(zip(truth, u)):
        for i in range(0, 3):
            imp = axs[i, j].imshow(truth_[i].T, aspect='auto')
            axs[i, j].axvline(x=seq_len, ymin=0, ymax=256, c='w', linestyle='--')
            axs[i+3, j].imshow(u_[i].detach().cpu().numpy().T, aspect='auto')
            axs[i+3, j].axvline(x=seq_len, ymin=0, ymax=256, c='w', linestyle='--')

    axs[0, 0].set_ylabel("Truth (y)")
    axs[1, 0].set_ylabel("Truth (y)")
    axs[2, 0].set_ylabel("Truth (y)")
    axs[3, 0].set_ylabel("SLT (y)")
    axs[4, 0].set_ylabel("SLT (y)")
    axs[5, 0].set_ylabel("SLT (y)")

    axs[-1, 0].set_xlabel("t")
    axs[-1, 1].set_xlabel("t")
    axs[-1, 2].set_xlabel("t")
    axs[-1, 3].set_xlabel("t")

    axs[0, 0].set_title(r'$ \beta=0.46$')
    axs[0, 1].set_title(r'$ \beta=0.75%')
    axs[0, 2].set_title(r'$ \beta=1.65$')
    axs[0, 3].set_title(r'$ \beta=2.55$')
    
    fig.colorbar(imp, ax=axs.ravel().tolist(), shrink=0.465, pad=0.015)
    plt.show()