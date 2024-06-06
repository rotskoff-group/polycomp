"""
Training a TFNO on Darcy-Flow
=============================

In this example, we demonstrate how to use the small Darcy-Flow example we ship with the package
to train a Tensorized Fourier-Neural Operator
"""

# %%
# 


import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from neuralop.models import TFNO
from neuralop import Trainer
from neuralop.training import CheckpointCallback
from AB_import import load_darcy_flow_small_local
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
from neuralop.training.callbacks import BasicLoggerCallback

device = 'cuda'


# %%
# Loading the Navier-Stokes dataset in 128x128 resolution
train_loader, test_loaders, data_processor = load_darcy_flow_small_local(
        n_train=1000, batch_size=32, 
        test_resolutions=[32], n_tests=[50],
        test_batch_sizes=[32],
        positional_encoding=False
)

data_processor = data_processor.to(device)

#data = data_processor.preprocess(data, batched=False)
train_samples = train_loader.dataset
data = train_samples[0]
data = data_processor.preprocess(data, batched=False)

# %%
# We create a tensorized FNO model

model = TFNO(in_channels=4, out_channels=2, n_modes=(32, 32), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
model = model.to(device)
model.load_checkpoint("/home/emmit/Rotskoff/neurop_data/wandb/run-20240603_152405-5a2kjacv/checkpoints", "model")

test_samples = test_loaders[32].dataset

fig = plt.figure(figsize=(12, 7))
where = [1,14,35]
for index in range(3):
    data = test_samples[where[index]]
    data = data_processor.preprocess(data, batched=False)
    # Input x
    x = data['x']
    # Model prediction
    out = model(x.unsqueeze(0))
    out_hold = torch.clone(out)
    data_processor.train = False
    #print(data_processor.train)
    print(out.shape)
    print(torch.mean(out[0,1]))
    out, data = data_processor.postprocess(out, data)
    print(torch.mean(out[0,1]))
    exit()
    #print(torch.allclose(out_hold, out, atol=1e-6))
    #print(data_processor.out_normalizer.mean)
    #print(data_processor.out_normalizer.std)
    #print(data['x'])
    #exit()
    # Ground-truth
    y = data['y']

    ax = fig.add_subplot(3, 5, index*5 + 1)
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    cax = ax.imshow(x[0], cmap='gray')
    fig.colorbar(cax, ax=ax)
    if index == 0: 
        ax.set_title('Input Field 1')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 5, index*5 + 2)
    cax = ax.imshow(y[0,0].squeeze().real)
    fig.colorbar(cax, ax=ax)
    if index == 0: 
        ax.set_title('Ground-truth Density A')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 5, index*5 + 3)
    cax = ax.imshow(out[0,0].squeeze().detach().cpu().numpy().real)
    fig.colorbar(cax, ax=ax)
    if index == 0: 
        ax.set_title('Model prediction Density A')
    plt.xticks([], [])
    plt.yticks([], [])

    n_out = out.detach().cpu().numpy()
    ax = fig.add_subplot(3, 5, index*5 + 4)
    cax = ax.imshow(y[0,1].squeeze().real)
    print(np.average(y[0,1]))
    fig.colorbar(cax, ax=ax)
    if index == 0: 
        ax.set_title('Ground-truth Density B')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 5, index*5 + 5)
    cax = ax.imshow(out[0,1].squeeze().detach().cpu().numpy().real)
    print(torch.mean(out[0,1]))
    fig.colorbar(cax, ax=ax)
    if index == 0: 
        ax.set_title('Model prediction Density B')
    plt.xticks([], [])
    plt.yticks([], [])


fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
plt.show()
