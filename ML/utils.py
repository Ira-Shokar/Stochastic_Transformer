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


import torch, numpy as np
import plot_utils

def load_truth():
        beta_0_3 = np.genfromtxt(f'/home/is500/Documents/QGF/plots/csv/umean_SL_20_0.3.csv', delimiter=',')[2840:2840+510]
        beta_0_6 = np.genfromtxt(f'/home/is500/Documents/QGF/plots/csv/umean_SL_20_0.6.csv', delimiter=',')[900:1410]
        beta_0_9 = np.genfromtxt(f'/home/is500/Documents/QGF/plots/csv/umean_SL_20_0.9.csv', delimiter=',').T[460:460+510]
        beta_1_8 = np.genfromtxt(f'/home/is500/Documents/QGF/plots/csv/umean_SL_20_1.8.csv', delimiter=',')[530:1040]
        return [beta_0_3, beta_0_6,  beta_0_9, beta_1_8]
        

    def load_truth_int():

        def load(seed, beta, t):
            arr = []
            u1 = np.genfromtxt(f'/home/is500/Documents/QGF/plots/csv/umean_SL_{seed}_{beta}.csv', delimiter=',')[t-10:t+500] 
            arr.append(u1)
            for i in range(100, 120, 10):
                u = np.genfromtxt(f'/home/is500/Documents/QGF/plots/csv/umean_SL_{seed}_{i}_{beta}_{t}.csv', delimiter=',')[:500]
                arr.append(np.concatenate((u1[:10] , u), axis=0))
            return arr

        arr   = []
        seeds = [50, 40, 40, 50]
        betas = [0.45, 0.75, 1.65, 2.55]
        times = [1100, 250, 310, 220]
        for i in range(4):
            arr.append(load(seeds[i], betas[i], times[i]))

        return arr


    def predict( Trans, truth, feat_dim, evolution_time, seq_len, ens_size=4, beta=0.3):

        truth = torch.Tensor(truth)

        z = torch.zeros((ens_size, seq_len+evolution_time, feat_dim)).to(utils.device)

        beta = torch.ones(ens_size, seq_len, 1).to(utils.device) * beta

        z[:, :seq_len] = torch.tensor(truth[:seq_len]).to(utils.device)

        for t in range(seq_len, evolution_time+seq_len):
           z[:, t], _ = Trans(torch.cat((z[:, t-seq_len:t], beta), dim=-1))

        return z

    def generate_plots(model, truth, truth_int, seq_len):

        u_arr     = []
        H_t_arr   = []
        H_p_arr   = []
        edges_arr = []

        for (truth, beta) in zip(truth, [0.3, 0.6, 0.9, 1.8]):
            u = predict(model, truth, 500, seq_len, 4, beta=beta)
            u_arr.append(u)

            truth_mat, preds_mat = calculate_grad_fields(torch.Tensor(truth).unsqueeze(0), u[0].unsqueeze(0))
            H_t, H_p, edges, _   = calculate_pdfs(truth_mat, preds_mat)
            H_t_arr.append(H_t)
            H_p_arr.append(H_p)
            edges_arr.append(edges)

        img    = plot_utils.plot_ens(truth, u_arr, seq_len)
        pdf    = plot_utils.plot_pdf(H_t_arr, H_p_arr, edges_arr)
        pdf_1d = plot_utils.plot_1d_pdf(H_t_arr, H_p_arr, edges_arr)

        energy_spec = plot_utils.plot_spectra(truth, u_arr, [0.3, 0.6, 0.9, 1.8])

        u_arr     = []
        H_t_arr   = []
        H_p_arr   = []
        edges_arr = []

        for (truth, beta) in zip(truth_int, [0.45, 0.75, 1.65, 2.55]):

            u = predict(model, truth[0], 500, seq_len, 3, beta=beta)
            u_arr.append(u)

            truth_mat, preds_mat = plot_utils.calculate_grad_fields(torch.Tensor(truth[0]).unsqueeze(0), u[0].unsqueeze(0))
            H_t, H_p, edges, _   = plot_utils.calculate_pdfs(truth_mat, preds_mat)
            H_t_arr.append(H_t)
            H_p_arr.append(H_p)
            edges_arr.append(edges)

        img_int    = plot_utils.plot_ens_int(truth_int, u_arr, seq_len)
        pdf_int    = plot_utils.plot_pdf(H_t_arr, H_p_arr, edges_arr)
        pdf_1d_int = plot_utils.plot_1d_pdf(H_t_arr, H_p_arr, edges_arr)

        energy_spec_int = plot_utils.plot_spectra_int(truth_int, u_arr, beta= [0.45, 0.75, 1.65, 2.55])

        return img, pdf, pdf_1d, energy_spec, img_int, pdf_int, pdf_1d_int, energy_spec_int


    @torch.no_grad()
    def calculate_grad_fields(truth_ens, preds_ens):

        preds_ens_size, evolution_time, y_size = preds_ens.size()

        # ensmeble, time, lat, (u, dy, dt)
        truth_mat = torch.zeros((preds_ens_size, evolution_time, y_size, 3))
        preds_mat = torch.zeros((preds_ens_size, evolution_time, y_size, 3))
        
        for j in range(preds_ens_size):
            truth_mat[j, :, :, 0] = truth_ens[j]
            preds_mat[j, :, :, 0] = preds_ens[j]

            for t in range(evolution_time):
                truth_mat[j, t, :, 1] = torch.gradient(truth_ens[j, t], spacing = 2.0)[0]
                preds_mat[j, t, :, 1] = torch.gradient(preds_ens[j, t], spacing = 2.0)[0]

            for y in range(y_size):
                truth_mat[j, :, y, 2] = torch.gradient(truth_ens[j, :, y], spacing = 2.0)[0]
                preds_mat[j, :, y, 2] = torch.gradient(preds_ens[j, :, y], spacing = 2.0)[0]

        return truth_mat, preds_mat 

    @torch.no_grad()
    def calculate_pdfs(truth_mat, preds_mat, nbins=100):

        u_t = truth_mat.flatten(0, 2)
        u_p = preds_mat.flatten(0, 2)

        p, edges = torch.histogramdd(u_t, nbins)

        edges_  = [edges[0].min(), edges[0].max(),
                edges[1].min(), edges[1].max(),
                edges[2].min(), edges[2].max()]
        range_  = [float(i.detach().cpu().numpy()) for i in edges_]

        q, _, = torch.histogramdd(u_p, nbins, range=range_)

        p /= p.sum()
        q /= q.sum()

        hellinger  = torch.sqrt(0.5*((torch.sqrt(p[0]) - torch.sqrt(q[0]))**2).sum())
        hellinger += torch.sqrt(0.5*((torch.sqrt(p[1]) - torch.sqrt(q[1]))**2).sum())
        hellinger += torch.sqrt(0.5*((torch.sqrt(p[2]) - torch.sqrt(q[2]))**2).sum())
        hellinger /= 3

        return p, q, edges, hellinger.item()


class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, X, Y, steps=None, batch_size=32, seq_forward=1, shuffle=True):
        self.X           = torch.tensor(X, dtype=torch.float32)
        self.Y           = torch.tensor(Y, dtype=torch.float32)
        self.seq_forward = seq_forward
        self.batch_size  = batch_size
        self.steps       = steps
        self.indexes     = np.arange(len(X))

        if shuffle: np.random.shuffle(self.indexes)

    def __len__(self): return self.steps

    def __getitem__(self, idx):
        X_batch = self.X[idx*self.batch_size: (idx+1)*self.batch_size]
        Y_batch = self.Y[idx*self.batch_size: (idx+1)*self.batch_size]
        return (X_batch, Y_batch)
