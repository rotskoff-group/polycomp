import torch 
import cupy as cp 

from polycomp.ft_system import * 
from neuralop.models import TFNO

import time



import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import TFNO
from neuralop import Trainer
from neuralop.training import CheckpointCallback
#from AB_import import load_darcy_flow_small_local
#from neuralop.utils import count_model_params
#from neuralop import LpLoss, H1Loss
from neuralop.training.callbacks import BasicLoggerCallback


from pathlib import Path


def create_ansatz(ps, model_dict):
    where_data = "/home/emmit/Rotskoff/neurop_data/clean_runs/wandb/run-20240606_150402-e2ikiizn/checkpoints/"
    device = 'cuda'

    if not (set(model_dict.keys()) == set(ps.poly_dict.keys())):
        raise ValueError("The model dictionary doesn't match the polymer dictionary")
    if not all(isinstance(value, str) for value in model_dict.values()): 
        raise ValueError("Not all values are strings")

    ps.nansatz_dict = {}
    for key in model_dict.keys():
        #model = TFNO(in_channels=4, out_channels=2, n_modes=(32, 32), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
        model = TFNO(in_channels=5, out_channels=3, n_modes=(32, 32), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
        model = model.to(device)
        model.load_checkpoint(model_dict[key], "model")
        data_processor = torch.load(model_dict[key] + "data_processor.pt")
        ps.nansatz_dict[key] = (model, data_processor)
##    model.load_checkpoint("/home/emmit/Rotskoff/neurop_data/wandb/run-20240603_152405-5a2kjacv/checkpoints", "model")

#    ps.data_processor = torch.load(where_data + "data_processor.pt")
#    ps.nansatz = model
    return
#    return model

def infer(ps):
#    t0 = time.time()
    ps.phi_all *= 0 
    for poly in ps.nansatz_dict.keys():
        if ps.poly_dict[poly]==0:
            continue
        #data = data1["x"][0].type(torch.float32).clone().to(device='cuda')
        data = cp.concatenate((ps.w_all, ps.grid.grid), axis=0)
        data = data.real.astype(cp.float32)
        data = torch.as_tensor(data, device = 'cuda')
        out = ps.nansatz_dict[poly][0](data.unsqueeze(0)) # ~85% of time
    #    out, stupid = data_processor.postprocess(out, data1)
        out = ps.nansatz_dict[poly][1].out_normalizer.inverse_transform(out)
        #print("Means", torch.mean(out[0], dim=(1,2)))
        #print("Mins", torch.amin(out[0], dim=(1,2)))
    #    print(torch.mean(out[0,1]))
        out = cp.asarray(out.squeeze().detach())
        out = out.astype(cp.complex128)
        for mon in set(poly.struct.tolist()):
            amt = poly.mon_mass[mon] * ps.poly_dict[poly]
            index = ps.monomers.index(mon)
            ps.phi_all[index] += out[index] / cp.average(out[index]) * amt # ~5% of time
            #TODO: maks sure this is definitely right

#        hold_avg = cp.average(ps.phi_all, axis=(1,2))
#        ps.phi_all = out.astype(cp.complex128)
#        ps.phi_all = (ps.phi_all.T * hold_avg / cp.average(ps.phi_all, axis=(1,2))).T
#    print(cp.mean(ps.phi_all.real, axis=(1,2)))
#    print(cp.amin(ps.phi_all.real, axis=(1,2)))

#    print("Pass")
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


