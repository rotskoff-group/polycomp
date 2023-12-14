import sys
sys.path.insert(0, "/scratch/users/epert/polycomp/gpu_polycomp")
from celluloid import Camera
import cupy as cp
import math
import matplotlib.pyplot as plt
import soft_exp_polymer as p
import se_gibbs as g_e
from mpl_toolkits.axes_grid1 import make_axes_locatable


cp.random.seed(1)

grid_spec = (32,32,32)
#grid_spec = (230,230)
#grid_spec = (1000,1000)
#grid_spec = (300,300)
V1 = 200
V2 = 200
box_length_1 = (V1**(1/3),V1**(1/3),V1**(1/3))
box_length_2 = (V2**(1/3),V2**(1/3),V2**(1/3))
#box_length_1 = (V1**(1/2),V1**(1/2))
#box_length_2 = (V2**(1/2),V2**(1/2))
grid = p.Grid(box_length=box_length_1, grid_spec = grid_spec)
grid_2 = p.Grid(box_length=box_length_2, grid_spec = grid_spec)

A_mon = p.Monomer("A", 1.)
B_mon = p.Monomer("B", -1.)
S_mon = p.Monomer("S", 0)
monomers = [B_mon, A_mon]

corr = 1
tot = 1
diff = 0
FH_terms = {
        frozenset({A_mon}) : tot, 
        frozenset({B_mon}) : tot, 
        frozenset({A_mon, B_mon}) : tot + diff, 
        }

N = 1

for key in FH_terms.keys():
    FH_terms[key] = FH_terms[key] * 1

AB_poly = p.Polymer("A", N, [(A_mon, 1), (B_mon, 1)])
polymers = [AB_poly]
spec_dict = {
        AB_poly : 0.01,
        }
spec_dict_2 = {
        AB_poly : 0.4,
        }
        
smear = 0.2

ps = p.PolymerSystem(monomers, polymers, spec_dict, FH_terms,
        grid, smear, salt_conc=0.0 * N, integration_width = 1/100)

ps.update_normal_from_density()
ps.update_density_from_normal()


relax_rates = cp.array([2 * 2 * corr * ps.grid.dV]*(ps.w_all.shape[0])) 
relax_temps = cp.array([0.001]*(ps.w_all.shape[0])) 
#relax_temps *= ps.gamma.real
psi_rate = 20 * ps.grid.dV 
psi_temp = 1
E = 400 
ps.update_normal_from_density()
ps.update_density_from_normal()
integrator_1 = p.CL_RK2(ps, relax_rates, relax_temps, psi_rate, psi_temp, E)


gibbs_time = 0.03
volume_time = 0.2
gibbs_sys = g_e.GibbsEnsemble(ps, integrator_1, gibbs_time, volume_time, spec_dict_2 = spec_dict_2, grid_2 = grid_2)
gibbs_sys.part_2.get_densities()
gibbs_sys.burn(500)
traj = []
vol_traj = []
for i in range(1000):
    gibbs_sys.burn(20)
    gibbs_sys.sample_pot(40, 1)
    gibbs_sys.gibbs_step()
    print(i)
    #traj.append(cp.concatenate((gibbs_sys.C_1 / 2, gibbs_sys.C_2 / 2, 
    traj.append(cp.concatenate((gibbs_sys.C_1, gibbs_sys.C_2, 
                                gibbs_sys.sampled_pot_1, gibbs_sys.sampled_pot_2)))
    vol_traj.append((gibbs_sys.part_1.grid.V, gibbs_sys.part_2.grid.V, 
                                    gibbs_sys.sampled_pressure_1, gibbs_sys.sampled_pressure_2))
    if i % 1 == 0:
        cp.savetxt('traj.npy', cp.array(traj))
        cp.savetxt('vol_traj.npy', cp.array(vol_traj))
#        print(traj)
#        print(vol_traj)
#    if i==300:
#        gibbs_time = 0.01
#    if i==600:
