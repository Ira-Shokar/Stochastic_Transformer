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

import numpy as np, torch, utils
from ST import ST_1D

def pretrain_data():  

    # Load data
    print('\nLoading Training Data. \n');
    X = np.load(f"{DATA_PATH}X_train_{SEQ_LENGTH}.npy")
    Y = np.load(f"{DATA_PATH}Y_train_{SEQ_LENGTH}.npy")
    print('Training Data Loaded. Number of Data Points = {}\n'.format(len(X)));

    X = utils.un_normalise(X)
    Y = utils.un_normalise(Y)

    val   = 200000
    Val_X = torch.tensor(X[val:], dtype=torch.float32, device=utils.device)
    Val_Y = torch.tensor(Y[val:], dtype=torch.float32, device=utils.device)
    X     = torch.tensor(X[:val], dtype=torch.float32, device=utils.device)
    Y     = torch.tensor(Y[:val], dtype=torch.float32, device=utils.device)
    X, Y  = utils.shuffle(X, Y) #type: ignore

    beta  = torch.ones(X.size(0), SEQ_LENGTH, 1, device=utils.device) * 0.9
    X     = torch.cat([X, beta], dim=-1)

    beta  = torch.ones(Val_X.size(0), SEQ_LENGTH, 1, device=utils.device) * 0.9
    Val_X = torch.cat([Val_X, beta], dim=-1)

def finetune_data():
    DATA_PATH = '/home/is500/Documents/transfer_learning/data/training_data/'

    print('Loading Data')
    X     = np.load(f"{DATA_PATH}X_train_paper_5000_{SEQ_LENGTH}.npy")
    Y     = np.load(f"{DATA_PATH}Y_train_paper_5000_{SEQ_LENGTH}.npy")
    Val_X = np.load(f"{DATA_PATH}Val_X_train_paper_5000_{SEQ_LENGTH}.npy")
    Val_Y = np.load(f"{DATA_PATH}Val_Y_train_paper_5000_{SEQ_LENGTH}.npy")

    print(f'Data Loaded - {len(X)} training examples')

def train(X, Y, Val_X, Val_Y):
    TRAINING_STEPS = len(X)     // BATCH_SIZE
    VAL_STEPS      = len(Val_X) // BATCH_SIZE
    training_set   = utils.DataGenerator(X    , Y    , TRAINING_STEPS, BATCH_SIZE)
    validation_set = utils.DataGenerator(Val_X, Val_Y, VAL_STEPS     , BATCH_SIZE)
    
    model = Stochastic_Latent_Transformer(
        epochs         = EPOCHS         ,
        seq_len        = SEQ_LENGTH     ,
        ens_size       = ENS_SIZE       ,
        latent_dim     = LATENT_DIM     ,
        learning_rate  = LEARNING_RATE  ,
        training_steps = TRAINING_STEPS ,
        val_steps      = VAL_STEPS      ,
        save_path      = SAVE_PATH      ,
    )

    model.fit(training_set, validation_set)  


if __name__ == '__main__':

    BATCH_SIZE    = 256;
    EPOCHS        = 500;
    LEARNING_RATE = 5e-4;
    LATENT_DIM    = 256; 
    SEQ_LENGTH    = 2;
    ENS_SIZE      = 2;

    #data = pretrain_data()
    data = finetune_data()
    train(*data)