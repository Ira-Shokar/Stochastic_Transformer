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
import models

class ST_1D:
    def __init__(self, feat_dim=256, latent_dim=256, seq_len=1, ens_size=4, epochs=1,
                 learning_rate=1e-4, training_steps=1, val_steps=1, num_heads=16, 
                 layers=2, width=2, save_path=None, file_name=None):
        super().__init__()

        # Define Parameters
        self.save_path      = save_path
        self.file_name      = file_name   

        self.total_epochs   = epochs
        self.training_steps = training_steps
        self.val_steps      = val_steps
        self.lr             = learning_rate

        self.feat_dim       = feat_dim 
        self.latent_dim     = latent_dim
        self.seq_len        = seq_len
        self.ens_size       = ens_size
        self.num_heads      = num_heads
        self.layers         = layers
        self.width          = width

        # Define Models
        self.model = models.Stochastic_Transformer_1D(self.latent_dim, self.seq_len).to(utils.device)

        self.load_model()

        # Define Optimiser
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Define LR scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser, factor=0.5, patience=10, threshold=1e-3, min_lr=self.lr*2e-3)

        self.truth_ensemble = utils.truth_ensemble(
            seq_len         = SEQ_LENGTH,
            ens_size        = 4,
            evolution_time  = 500,
            time_step_start = 100)

        self.truth_long = utils.truth_long(seq_len=SEQ_LENGTH)

        # Loss Tracker
        self.loss_dict     = {'MAE':0, 'rep':0, 'spec':0}
        self.val_loss_dict = {'MAE':0, 'rep':0, 'spec':0}


    def fit(self, data, val_data):

        for self.epoch in range(self.total_epochs):

            with tqdm.trange(self.training_steps, ncols=140) as pbar:
                self.loss_dict     = {x:0 for x in self.loss_dict}
                self.val_loss_dict = {x:0 for x in self.val_loss_dict}
               
                self.model.train()
                if torch.cuda.device_count() > 1: self.model = torch.nn.DataParallel(self.model)
                self.model = self.model.to(utils.device)

                for self.step, train_batch in zip(pbar, data):
                    self.forward(*train_batch, train=True)
                    if self.step%100==0: self.track_losses(pbar)

                    if self.step==(self.training_steps-1):
                        
                        self.model.eval()
                        for self.val_step, val_batch in zip(range(self.val_steps), val_data):
                            with torch.no_grad():
                                self.forward(*val_batch)

                                if self.val_step==(self.val_steps-1):
                                    if torch.cuda.device_count() > 1: self.model = self.model.module
                                    if self.epoch%5==0: self.generate_plots()
                                    self.track_losses(pbar, val=True)
                                    self.scheduler.step(self.val_loss_dict['MAE'])

        self.save_model(self.model, self.epoch, self.total_epochs, self.run_num, self.save_path, final=True)

    def CRPS(self, x, y, p=2):

        while y.dim()!=x.dim(): y = y.unsqueeze(1)

        MSE      = torch.cdist(x, y, p).mean() / x.size(-1)**0.5 # norm by sqrt of latent dim
        ens_var  = torch.cdist(x, x, p).mean() / x.size(-1)**0.5 # norm by sqrt of latent dim
        ens_var *= self.ens_size/(self.ens_size - 1) # to result in 1/[m(m-1)]

        return MSE-0.5*ens_var, MSE, ens_var

    def MAE_spectral(self, x, y, p=2):
        x = torch.fft.rfft(x, norm='forward').abs()
        y = torch.fft.rfft(y, norm='forward').abs()
        return torch.nn.functional.l1_loss(x, y, reduction='mean')

    def forward(self, x, y, train=False): 
        
        x = x.to(utils.device)
        y = y.to(utils.device)

        # Forward Pass through Model for m ensemble members
        o = torch.stack([self.model(x)[0] for i in range(self.ens_size)], dim=1)

        crps, mae, rep = self.CRPS(o, y)         # MAE and Ens Var Terms
        spec           = self.MAE_spectral(o, y) # Spectral MAE

        if train: # Backward Pass
            (crps + spec).backward()
            self.optimiser.step()
            self.optimiser.zero_grad()

        # Update Metric Tracker
        loss_dict = self.loss_dict if train else self.val_loss_dict
                
        loss_dict['MAE']  += mae.item()
        loss_dict['rep']  += rep.item()
        loss_dict['spec'] += spec.item()

    def save_model(self, final=False):
        file_name  = f'{self.latent_dim}_{self.total_epochs}_{RUN_NUM}'
        if not final: file_name += f'_{self.epoch}'
        torch.save(self.model.state_dict(), f"{self.save_path}weights/weights_Trans_{file_name}.pt")

    def load_model(self):
        weights_path = f'/home/is500/Documents/Beta_Plane_Jets/data/outputs/AE_Transformer/weights/'
        file_name    = f'weights_{"RNN"}_{256}_{500}_{10005}' #400
        self.model.load_state_dict(torch.load(f'{weights_path}{file_name}.pt', map_location='cpu'))


    def track_losses(self, pbar, val=False):
        
        loss_dict = {x:self.loss_dict[x]/(self.step+1) for x in self.loss_dict}
        if val==False:
            pbar.set_postfix({
                'epoch'   : f"{self.epoch}/{self.total_epochs}",
                'MAE'     : f"{loss_dict['MAE']  :.2E}",
                'MAE_val' : f"---------",
                'spec'    : f"{loss_dict['spec'] :.2E}" ,
                'spec_val': f"---------",
            })
        
        else:
            val_loss_dict = {x:self.val_loss_dict[x]/(self.val_step+1) for x in self.val_loss_dict}

            pbar.set_postfix({
                'epoch'    : f"{self.epoch}/{self.total_epochs}",
                'MAE'      : f"{loss_dict['MAE']      :.2E}" ,
                'MAE_val'  : f"{val_loss_dict['MAE']  :.2E}" ,
                'spec'     : f"{loss_dict['spec']     :.2E}" ,
                'spec_val' : f"{val_loss_dict['spec'] :.2E}" ,
                })
