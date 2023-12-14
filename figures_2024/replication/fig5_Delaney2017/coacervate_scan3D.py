import sys
sys.path.insert(0, "/scratch/users/epert/polycomp/gpu_polycomp")
from celluloid import Camera
import cupy as cp
import math
import matplotlib.pyplot as plt
import soft_exp_polymer as p
from mpl_toolkits.axes_grid1 import make_axes_locatable


CA_amt = float(sys.argv[1])
cp.random.seed(int(CA_amt * 10000000))

grid_spec = (32,32,32)
#grid_spec = (300,300)
box_length = (6.4,6.4,6.4)
grid = p.Grid(box_length=box_length, grid_spec = grid_spec)

A_mon = p.Monomer("A", 1.)
B_mon = p.Monomer("B", -1.)
monomers = [B_mon, A_mon]

tot = 1
FH_terms = {
        frozenset({A_mon}) : tot, 
        frozenset({B_mon}) : tot, 
        frozenset({A_mon, B_mon}) : tot, 
        }

N = 80
N2 = 80

for key in FH_terms.keys():
    FH_terms[key] = FH_terms[key] * 1

AB_poly = p.Polymer("AB", N, [(A_mon, 1), (B_mon, 1)])
polymers = [AB_poly]
spec_dict = {
        AB_poly : CA_amt / 2,
        }

smear = 0.163299316185545
smear = 0.2
ps = p.PolymerSystem(monomers, polymers, spec_dict, FH_terms,
        grid, smear, salt_conc=0.0 * N, integration_width = 1/100)



relax_rates = cp.array([2 * ps.grid.dV]*(ps.w_all.shape[0])) 
print(relax_rates)
relax_temps = cp.array([1 + 0j]*(ps.w_all.shape[0])) 
#relax_temps *= ps.gamma
psi_rate = 20 * ps.grid.dV 
print(psi_rate)
psi_temp = 1
E = 400
integrator_1 = p.CL_RK2(ps, relax_rates, relax_temps, psi_rate, psi_temp, E)

print(ps.normal_modes)
print(ps.normal_evalues)

ps.update_density_from_normal()
ps.update_normal_from_density()
ps.get_densities()




free_en = []
pi_traj = []
chem_pot_traj = []
when = 0
steps = int(300 * 30 / 30)
for i in range(30):
    for _ in range(steps):
        integrator_1.ETD(for_pressure=True)
        free_en.append((when, ps.get_free_energy(E).get()))
#        tot_pi, pi_dens, phi_del = ps.get_pressure()
        tot_pi = ps.get_pressure()
        pi_traj.append((tot_pi.get()))
        ps.get_chemical_potential()
        chem_pot_traj.append((ps.chem_pot_dict[AB_poly].get()))
        when += 1
    cp.save('midpoint', ps.w_all)
    cp.save('midpoint_psi', ps.psi)
    print(i)



free_en_traj = cp.array(free_en)
pi_traj = cp.expand_dims(cp.array(pi_traj), axis=1)
chem_pot_traj = cp.expand_dims(cp.array(chem_pot_traj), axis=1)
full_traj = cp.concatenate((free_en_traj, pi_traj, chem_pot_traj), axis=1)
cp.savetxt("CA_traj3D/traj_%.10f" % CA_amt, full_traj)
print("CA_traj3D/traj_%.10f" % CA_amt)


