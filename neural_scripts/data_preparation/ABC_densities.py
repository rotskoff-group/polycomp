#This script takes all the input files and generates the density corresponding to the specified species 


from celluloid import Camera
import cupy as cp
import polycomp.ft_system as p
from polycomp.observables import * 
from polycomp.neural_ansatz import * 

import torch
import sys

if sys.argv[1] not in ['ABC', 'AB', 'AC', 'BC', 'ABCCBA', 'CAABBC']:
    raise ValueError("wrong key")
else:
     poly_key = sys.argv[1]

print("Attempting load")
cp_inputs = cp.load(sys.argv[2])

print("Loaded successfully")

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

big_hold = []
#Imports each field, recalibrates the fields and then computes the corresponding densities
for i in range(cp_inputs.shape[0]):
    if i % 500 == 0:
        print(i)
    dl = cp_inputs[i,-2:,1,1].real
    l = dl * cp.array(cp_inputs.shape[-2:]).astype(float)
    ps.grid.update_l(l)
    if not cp.allclose(ps.grid.grid, cp_inputs[i, -2:].real):
        raise ValueError("Something went wrong with matching the grids")


    ps.w_all.real = cp_inputs[i,:-2]
    ps.update_normal_from_density()
    ps.get_densities()
    hold = cp.stack((cp.copy(ps.phi_all[ps.monomers.index(A_mon)]), cp.copy(ps.phi_all[ps.monomers.index(B_mon)]), cp.copy(ps.phi_all[ps.monomers.index(C_mon)])))
    big_hold.append(cp.copy(hold))
cp_outputs = cp.array(big_hold)

#Reformat them to torch file format
torch_inputs = torch.as_tensor(cp_inputs, device = 'cuda')
torch_outputs = torch.as_tensor(cp_outputs, device = 'cuda')

torch_dict = {'x' : torch_inputs.cpu(),
          'y' : torch_outputs.cpu()}

torch.save(torch_dict, poly_key + "_data.pt")











