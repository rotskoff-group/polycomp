import sys
sys.path.insert(0, "..")
import cupy as cp
import soft_exp_polymer as p
from line_profiler import LineProfiler
from se_MDE import *

lp = LineProfiler()

cp.random.seed(0)

grid2d = p.Grid(box_length=(2,2), grid_spec = (20,20))

A_mon = p.Monomer("A", 1.)
B_mon = p.Monomer("B", -1.)
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

grid_spec = (512,512)
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
ps.get_densities()

lp.add_function(integrator.ETD)
lp.add_function(ps.get_densities)
lp.add_function(integrate_s)
lp.add_function(s_step)
lp.add_function(ps.convolve)
#lp_wrapper = lp(integrator.ETD)
#lp_wrap_2 = lp(ps.get_densities)
#lp_wrap_3 = lp(integrate_struct)
ps.get_densities()
for i in range(30):
    print(i)
    lp.run('integrator.ETD()')
#    lp_wrapper()
#    lp_wrap_2()
#    lp_wrap_3(A_poly.struct, A_poly.h_struct, )
ps.get_densities()
lp.print_stats()
