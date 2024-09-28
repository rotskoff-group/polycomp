from celluloid import Camera
import cupy as cp
import polycomp.ft_system as p
from polycomp.observables import * 
from polycomp.neural_ansatz import * 

import torch
import sys

#Set a seed for reproducibility (or turn off for full randomization)
#cp.random.seed(14)

if sys.argv[1] not in ['ABC', 'AB', 'AC', 'BC', 'ABCCBA', 'CAABBC']:
    raise ValueError("wrong key")
else:
     poly_key = sys.argv[1]

#Declare all of your polymers with their name and charge
A_mon = p.Monomer("A", 0)
B_mon = p.Monomer("B", 0)
C_mon = p.Monomer("C", 0)

#Declare a list of all the monomer types in the simulation
#(Salts are handled automatically if needed)
monomers = [A_mon, B_mon, C_mon]

tot = 130
diff = 21

#Declare a Flory-Huggins array with each cross iteraction
#Here we use total and difference to simplify the description
FH_terms = {
        frozenset({A_mon}) : tot, 
        frozenset({B_mon}) : tot, 
        frozenset({C_mon}) : tot, 
        frozenset({A_mon, B_mon}) : tot + diff, 
        frozenset({A_mon, C_mon}) : tot + diff, 
        frozenset({B_mon, C_mon}) : tot + diff, 
        }

#Declare the reference polymer length for the system. In this case it will be the
# same as the length of the only polymer in solution 
N = 5

#Declare all the polymer types in solution. In this case we have a single "AB" diblock
# copolymer that is half A and half B, with a total length of N.
poly_options =  {
'ABC' : (p.Polymer("ABC", N, [(A_mon, 1.0/3), (B_mon, 1.0/3), (C_mon, 1.0/3)]), 1), 
'ABCCBA' : (p.Polymer("ABC", N, [(A_mon, 1.0/3), (B_mon, 1.0/3), (C_mon, 2.0/3), (B_mon, 1.0/3), (A_mon, 1.0/3)]), 2),
'CAABBC' : (p.Polymer("ABC", N, [(C_mon, 1.0/3), (A_mon, 2.0/3), (B_mon, 2.0/3), (C_mon, 1.0/3)]), 2),
'AB' : (p.Polymer("ABC", N, [(A_mon, 1.0/3), (B_mon, 1.0/3)]), 2.0/3),
'AC' : (p.Polymer("ABC", N, [(A_mon, 1.0/3), (C_mon, 1.0/3)]), 2.0/3),
'BC' : (p.Polymer("ABC", N, [(B_mon, 1.0/3), (C_mon, 1.0/3)]), 2.0/3),
}

ABC_poly = poly_options[poly_key][0]
amt = 1 / poly_options[poly_key][1]

print(ABC_poly, amt)

#Declare a list of all the polymers in simulation 
polymers = [ABC_poly]

#Declare a dictionary with all the species in the system (this will include polymers
# and solvents, but here we just have one polymer). We also declare the concentration
# of each species, in this case just 1. 
spec_dict = {
        ABC_poly : amt,
        }
#Declare the number of grid points across each axis. This will be a 2D simulation 
# with 256 grid points along each dimension. 
grid_spec = (128,128)

#Declare the side length of the box along each axis. Here we have 25x25 length square.
box_length = (16,16)

#Declare the grid object as specified using our parameterss.
grid = p.Grid(box_length=box_length, grid_spec = grid_spec)

#Declare the smearing length for the charge and density
smear = 0.2

#We can now declare the full polymer system. Read the full documentation for details, 
# but we use previously declared variables and specify salt concentration and 
# integration fineness along the polymer. 
ps = p.PolymerSystem(monomers, polymers, spec_dict, FH_terms,
        grid, smear, salt_conc=0.0 * N, integration_width = 1/20)

#Now we move to our integration parameters. We need a timestep associated with each 
# field, but they'll all be the same here. 
relax_rates = cp.array([0.5]*(ps.w_all.shape[0])) * 2

#We also declare a temperature array which is the same shape
temps = cp.array([0.001 + 0j]*(ps.w_all.shape[0]))

#This temperature corresponds to using the "standard" CL integrator, but other versions
# generally are valid
temps *= ps.gamma.real

#Because this is an uncharged system, we set E to 0, and set all the electric field 
# rates to 0 as well
E = 0
psi_rate = 0
psi_temp = 0

#Now we actually declare the integrator
integrator = p.CL_RK2(ps, relax_rates, temps, psi_rate, psi_temp, E)

#generate an initial density for the starting plot
ps.get_densities()

#Initial plots
#im[0][0] = axes[0].imshow(ps.phi_all[ps.monomers.index(A_mon)].real.get(), cmap = 'Blues')
#axes[0].set_title('A Dens')
#im[0][1] = axes[1].imshow(ps.w_all[0].real.get(), cmap = 'Reds')
#axes[1].set_title('B Dens')
#im[0][2] = axes[2].imshow(cp.sum(ps.phi_all, axis=0).real.get(), cmap = 'Greys')
#axes[2].set_title('Total density')


#Set the number of steps per frame
steps = 3000

f_traj = []
d_traj = []
#Set the number of arrays to capture
import time
t0 = time.time()
for i in range(200):

    #We average over multiple views to reduce the noise for visualization, could just 
    # plot directly as well to simplify this
    new_l = cp.random.uniform(8,18)
    ps.grid.update_l((new_l, new_l))
    print(grid.l)
    ps.w_all.real = cp.random.normal(loc=0.0, scale=0.001, size=ps.w_all.shape) * 0
    ps.w_all.imag *= 0
    ps.update_normal_from_density()
    integrator.ETD()
    for _ in range(steps):

        rando = cp.random.rand()
        #Collect the variables of interest every step and average over some of them
        if rando < 5.0 /steps:
            f_traj.append(cp.copy(cp.concatenate((ps.w_all, grid.grid), axis=0)))
        integrator.ETD()
#        if (_+1) % 200 == 0 :
#            d_traj.append(cp.stack((cp.copy(ps.phi_all[ps.monomers.index(A_mon)]), cp.copy(ps.phi_all[ps.monomers.index(B_mon)])), axis=0))

    #Print current progress
    if i % 1 ==0: 
        print(time.time() - t0)
        t0 = time.time()
        print(i)
print(f_traj)
f_traj = cp.stack(f_traj,axis=0)
#d_traj = cp.stack(d_traj,axis=0)

f_traj = cp.squeeze(f_traj)
print(f_traj.shape)
cp.save(poly_key + "_fields", f_traj)
#print(d_traj.shape)


f_traj_torch = torch.as_tensor(f_traj, device = 'cuda')
#d_traj_torch = torch.as_tensor(d_traj, device = 'cuda')

#output = {'x' : f_traj_torch.cpu(),
#          'y' : d_traj_torch.cpu()}

#torch.save(output, 'ABA_size1.pt')
exit()
