import torch 
import cupy as cp 

from polycomp.ft_system import * 
from neuralop.models import TFNO





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


from pathlib import Path

import time 

train_loader, test_loaders, data_processor = load_darcy_flow_small_local(
        n_train=1000, batch_size=32, 
        test_resolutions=[32], n_tests=[50],
        test_batch_sizes=[32],
        positional_encoding=False
)

data_processor = data_processor.to('cuda')
data_processor.train = False
data_path = './'
data1 = torch.load(
    Path(data_path).joinpath(f"AB_stripe2.pt").as_posix()
)
data1['y'] = data1['y'].to(device = 'cuda')
#    print(data1)


def create_ansatz(ps):

    device = 'cuda'
    model = TFNO(in_channels=4, out_channels=2, n_modes=(32, 32), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
    model = model.to(device)
#    model.load_checkpoint("/home/emmit/Rotskoff/neurop_data/clean_runs/wandb/run-20240604_180216-fotz8729/checkpoints", "model")
    model.load_checkpoint("/home/emmit/Rotskoff/neurop_data/wandb/run-20240603_152405-5a2kjacv/checkpoints", "model")

    ps.nansatz = model
    return model

def infer(ps):
#    t0 = time.time()

    data = data1["x"][0].type(torch.float32).clone().to(device='cuda')
    data = cp.concatenate((ps.w_all, ps.grid.grid), axis=0)
    data = data.real.astype(cp.float32)
    data = torch.as_tensor(data, device = 'cuda')

    out = ps.nansatz(data.unsqueeze(0))
    out, stupid = data_processor.postprocess(out, data1)

    #print("Means", torch.mean(out[0], dim=(1,2)))
    #print("Mins", torch.amin(out[0], dim=(1,2)))
#    print(torch.mean(out[0,1]))
    out = cp.asarray(out.squeeze().detach())
#    print(torch.mean(out[0,1]))

    hold_avg = cp.average(ps.phi_all, axis=(1,2))
    ps.phi_all = out.astype(cp.complex128)
    ps.phi_all = (ps.phi_all.T * hold_avg / cp.average(ps.phi_all, axis=(1,2))).T
#    print(cp.mean(ps.phi_all.real, axis=(1,2)))
#    print(cp.amin(ps.phi_all.real, axis=(1,2)))

#    print("Pass")
#    print(-t0+time.time())
    return

def self_contained():
    train_loader, test_loaders, data_processor = load_darcy_flow_small_local(
            n_train=1000, batch_size=32, 
            test_resolutions=[32], n_tests=[50],
            test_batch_sizes=[32],
            positional_encoding=False
    )

    from pathlib import Path
    data_path = './'
    data1 = torch.load(
        Path(data_path).joinpath(f"AB_stripe2.pt").as_posix()
    )
    print(data1)

    data = data1["x"][0].type(torch.float32).clone().to(device='cuda')
    
    print(data)
    print(data.shape)

    model = TFNO(in_channels=4, out_channels=2, n_modes=(32, 32), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
    model = model.to(device='cuda')
    model.load_checkpoint("/home/emmit/Rotskoff/neurop_data/wandb/run-20240603_151946-wg2jo1zo/checkpoints", "model")


    print(data)
    print(model)
    print(data.dtype)
    out = model(data.unsqueeze(0))
    print(out.shape)
    print("COMPLETED")
    exit()


