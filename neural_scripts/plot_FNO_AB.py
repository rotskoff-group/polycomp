"""
Training a TFNO on Darcy-Flow
=============================

In this example, we demonstrate how to use the small Darcy-Flow example we ship with the package
to train a Tensorized Fourier-Neural Operator
"""

# %%
# 


import torch
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
data = data_processor.preprocess(train_loader.dataset[0], batched=False)


print(train_loader.dataset[0]['x'].shape)
#data = data_processor.preprocess(data, batched=False)
train_samples = train_loader.dataset

# %%
# We create a tensorized FNO model

model = TFNO(in_channels=4, out_channels=2, n_modes=(32, 32), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
model = model.to(device)

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()


# %%
#Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=5e-3, 
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


# %%
# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}


# %%


print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()


wandb_args =  dict(
#        name="A_test",
        group="emmit-pert",
        project="warmup",
#        entity=config.wandb.entity,
    )

import wandb
logger = BasicLoggerCallback(wandb_args)
checkpoints = CheckpointCallback(save_dir=wandb.run.dir + "/../checkpoints", save_interval=1,
                                 save_optimizer=True, save_scheduler=True)
print(wandb.run.dir)

# %% 
# Create the trainer
trainer = Trainer(model=model, n_epochs=150,
                  device=device,
                  data_processor=data_processor,
                  wandb_log=True,
                  log_test_interval=1,
                  use_distributed=False,
                  log_output=True,
                  callbacks=[
                      logger, checkpoints
                      ],
                  verbose=True)


# %%
# Actually train the model on our small Darcy-Flow dataset
trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler, 
              regularizer=None, 
              training_loss=train_loss,
              eval_losses=eval_losses)


# %%
# Plot the prediction, and compare with the ground-truth 
# Note that we trained on a very small resolution for
# a very small number of epochs
# In practice, we would train at larger resolution, on many more samples.
# 
# However, for practicity, we created a minimal example that
# i) fits in just a few Mb of memory
# ii) can be trained quickly on CPU
#
# In practice we would train a Neural Operator on one or multiple GPUs

test_samples = test_loaders[32].dataset

fig = plt.figure(figsize=(12, 7))
where = [1,14,35]
for index in range(3):
    data = test_samples[where[index]]
    data = data_processor.preprocess(data, batched=False)
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y']
    # Model prediction
    out = model(x.unsqueeze(0))

    ax = fig.add_subplot(3, 5, index*5 + 1)
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    ax.imshow(x[0], cmap='gray')
    if index == 0: 
        ax.set_title('Input Field 1')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 5, index*5 + 2)
    ax.imshow(y[0,0].squeeze().real)
    if index == 0: 
        ax.set_title('Ground-truth Density A')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 5, index*5 + 3)
    ax.imshow(out[0,0].squeeze().detach().cpu().numpy().real)
    if index == 0: 
        ax.set_title('Model prediction Density A')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 5, index*5 + 4)
    ax.imshow(y[0,1].squeeze().real)
    if index == 0: 
        ax.set_title('Ground-truth Density B')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 5, index*5 + 5)
    ax.imshow(out[0,1].squeeze().detach().cpu().numpy().real)
    if index == 0: 
        ax.set_title('Model prediction Density B')
    plt.xticks([], [])
    plt.yticks([], [])


fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
plt.show()
