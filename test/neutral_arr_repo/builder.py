import sys
sys.path.insert(0, "../..")
import cupy as cp
import soft_exp_polymer as p

cp.random.seed(0)

grid2d = p.Grid(box_length=(2,2), grid_spec = (20,20))

A_mon = p.Monomer("A", 0.)
B_mon = p.Monomer("B", -0.)
S_mon = p.Monomer("S", 0)
monomers = [A_mon, B_mon, S_mon]

        
FH_terms = {
        frozenset({A_mon}) : 2, 
        frozenset({B_mon}) : 2, 
        frozenset({S_mon}) : 2, 
        frozenset({A_mon, B_mon}) : 3, 
        frozenset({A_mon, S_mon}) : 3, 
        frozenset({B_mon, S_mon}) : 3, 
        }

N = 5

A_poly = p.Polymer("A", N, [(A_mon, 1)])
B_poly = p.Polymer("B", N, [(B_mon, 1)])
polymers = [A_poly, B_poly]
spec_dict = {
        A_poly : .3,
        B_poly : .3,
        S_mon : 1.5 * N
        }

grid_spec = (128,128)
box_length = (10,10)
grid = p.Grid(box_length=box_length, grid_spec = grid_spec)


smear = 0.1
ps = p.PolymerSystem(monomers, polymers, spec_dict, FH_terms,
        grid, smear, salt_conc=0.0 * N, integration_width = 1/30)



relax_rates = cp.array([1 * ps.grid.dV]*(ps.w_all.shape[0]))
w_temps = cp.array([0.003 + 0j]*(ps.w_all.shape[0]))
w_temps *= ps.gamma.imag
psi_rate = 2.5 * ps.grid.dV
psi_temp = 1
E = 10000
integrator = p.CL_RK2(ps, relax_rates, w_temps, psi_rate, psi_temp, E)

#ps.w_all = cp.random.rand(3,128,128) * 10 + 1j * cp.random.rand(3,128,128) * 10
ps.w_all = cp.load("random_w_all.npy")
ps.get_densities()
cp.save("phi_all", ps.phi_all)
integrator.ETD()
ps.get_densities()
cp.save("integrated_phi_all", ps.phi_all)
