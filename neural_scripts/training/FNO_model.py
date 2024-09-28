# This file was created as a modification of a file from the ae6bdb948b1733a8c1bb862de8127c55c97e3074 commit
# of the neuraloperator package, from April 10, 2024 and recognition should go to the developers of
# that package. Please reference their paper on arxiv at 2010.08895.

import time
import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import TFNO
from neuralop import Trainer
from neuralop.training import CheckpointCallback
from AB_import import load_custom_pt
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
from neuralop.training.callbacks import BasicLoggerCallback
import wandb

device = "cuda"

poly = sys.argv[1]
data = poly + "_data.pt"
test_train_split = 14.0 / 16

# %%
# Loading the dataset
train_loader, test_loaders, data_processor = load_custom_pt(
    data_name=data,
    test_train_split=test_train_split,
    batch_size=32,
    test_batch_sizes=[32],
    positional_encoding=False,
)

data_processor = data_processor.to(device)
data = data_processor.preprocess(train_loader.dataset[0], batched=False)


print(train_loader.dataset[0]["x"].shape)
train_samples = train_loader.dataset

# We create a tensorized FNO model
model = TFNO(
    in_channels=5,
    out_channels=3,
    n_modes=(32, 32),
    hidden_channels=32,
    projection_channels=64,
    factorization="tucker",
    rank=0.42,
)
model = model.to(device)

n_params = count_model_params(model)
print(f"\nOur model has {n_params} parameters.")
sys.stdout.flush()


# Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses = {"h1": h1loss, "l2": l2loss}
eval_losses = {"h1": h1loss}


# %%


print("\n### MODEL ###\n", model)
print("\n### OPTIMIZER ###\n", optimizer)
print("\n### SCHEDULER ###\n", scheduler)
print("\n### LOSSES ###")
print(f"\n * Train: {train_loss}")
print(f"\n * Test: {eval_losses}")
sys.stdout.flush()


wandb_args = dict(
    name=poly + "_model",
    group="emmit-pert",
    project="sherlock_ABC_long",
    id=poly + str(time.time()),
)

logger = BasicLoggerCallback(wandb_args)
checkpoints = CheckpointCallback(
    save_dir=wandb.run.dir + "/../checkpoints",
    save_interval=1,
    save_optimizer=True,
    save_scheduler=True,
)
print(wandb.run.dir)

# Create the trainer
trainer = Trainer(
    model=model,
    n_epochs=4530,
    device=device,
    data_processor=data_processor,
    wandb_log=True,
    log_test_interval=1,
    use_distributed=False,
    log_output=True,
    callbacks=[logger, checkpoints],
    verbose=True,
)


# Actually train the model
trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=None,
    training_loss=train_loss,
    eval_losses=eval_losses,
)


torch.save(data_processor, wandb.run.dir + "/../checkpoints/" + "data_processor.pt")

test_samples = test_loaders[32].dataset

# Plot some simple examples of inference and data
fig = plt.figure(figsize=(12, 7))
where = [1, 14, 35]
for index in range(3):
    data = test_samples[where[index]]
    data = data_processor.preprocess(data, batched=False)
    # Input x
    x = data["x"]
    # Ground-truth
    y = data["y"]
    # Model prediction
    out = model(x.unsqueeze(0))

    ax = fig.add_subplot(3, 5, index * 5 + 1)
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    ax.imshow(x[0], cmap="gray")
    if index == 0:
        ax.set_title("Input Field 1")
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 5, index * 5 + 2)
    ax.imshow(y[0, 0].squeeze().real)
    if index == 0:
        ax.set_title("Ground-truth Density A")
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 5, index * 5 + 3)
    ax.imshow(out[0, 0].squeeze().detach().cpu().numpy().real)
    if index == 0:
        ax.set_title("Model prediction Density A")
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 5, index * 5 + 4)
    ax.imshow(y[0, 1].squeeze().real)
    if index == 0:
        ax.set_title("Ground-truth Density B")
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 5, index * 5 + 5)
    ax.imshow(out[0, 1].squeeze().detach().cpu().numpy().real)
    if index == 0:
        ax.set_title("Model prediction Density B")
    plt.xticks([], [])
    plt.yticks([], [])


fig.suptitle("Inputs, ground-truth output and prediction.", y=0.98)
plt.tight_layout()
plt.savefig("plots/" + poly + "_fig")
# plt.show()
