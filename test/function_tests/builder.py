import sys
sys.path.insert(0, "../..")
import cupy as cp
import soft_exp_polymer as p
from se_MDE import * 

cp.random.seed(0)

grid2d = p.Grid(box_length=(2,2), grid_spec = (20,20))

A_mon = p.Monomer("A", 1.)
B_mon = p.Monomer("B", -1.)
monomers = [A_mon, B_mon]

        
FH_terms = {
        frozenset({A_mon}) : 2, 
        frozenset({B_mon}) : 2, 
        frozenset({A_mon, B_mon}) : 3, 
        }

N = 5

A_poly = p.Polymer("A", N, [(A_mon, 1)])
ABA_poly = p.Polymer("ABA", N, [(A_mon, 0.25), (B_mon, 0.5), (A_mon, 0.25)])
AB_poly = p.Polymer("AB", N, [(A_mon, 0.5), (B_mon, 0.5)])
ABAB_poly = p.Polymer("ABAB", N, [(A_mon, 0.1), (B_mon, 0.2), (A_mon, 0.3), (B_mon, 0.4)])
polymers = [A_poly, AB_poly, ABA_poly, ABAB_poly]
int_width = 1/30
for polymer in polymers:
    polymer.build_working_polymer(int_width, polymer.total_length / N)
    print(polymer.struct)
    print(polymer.h_struct)


grid_spec = (128,128)
box_length = (10,10)
grid = p.Grid(box_length=box_length, grid_spec = grid_spec)

#ps.w_all = cp.random.rand(3,128,128) * 10 + 1j * cp.random.rand(3,128,128) * 10
w_all = cp.load("random_w_all.npy")
psi = cp.load("random_psi.npy")

P_dict = {A_mon : w_all[0], B_mon : w_all[1]}
q_r = q_r_dag = cp.ones_like(w_all[0])
for poly in polymers:
    q_r_hold, q_r_dag_hold = integrate_s(poly.struct, poly.h_struct, P_dict, q_r, q_r_dag, grid)
    print(q_r_hold.shape)
    print(q_r_dag_hold.shape)
    cp.save(poly.name + "_q_r.npy", q_r_hold)
    cp.save(poly.name + "_q_r_dag.npy", q_r_dag_hold)

cp.save("one_step_A.npy", s_step(q_r, 1/30, w_all[0], grid))
