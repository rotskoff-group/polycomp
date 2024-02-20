import unittest
import cupy as cp
import polycomp.ft_system as p
from polycomp.mde import *

def build_polysystem(charge):
    cp.random.seed(0)

    grid2d = p.Grid(box_length=(2,2), grid_spec = (20,20))
    
    A_mon = p.Monomer("A", charge)
    B_mon = p.Monomer("B", -charge)
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
    
    return integrator, ps


class ChargedNumericTests(unittest.TestCase):
    def setUp(self):
        self.integrator, self.ps = build_polysystem(1.0)

    def test_get_density(self):
        #tests that the density for a random array from file of chemical potential
        # gives the correct matching density
        self.ps.w_all = cp.load("charged_arr_repo/random_w_all.npy")
        self.ps.psi = cp.load("charged_arr_repo/random_psi.npy")
        self.ps.get_densities()
        cp.testing.assert_allclose(cp.load("charged_arr_repo/phi_all.npy"), self.ps.phi_all, rtol=1e-13)


    def test_integrator(self):
        #checks that one step of integration gives the same results as previous after
        # random number has been reset
        cp.random.seed(0)
        self.ps.w_all = cp.load("charged_arr_repo/random_w_all.npy")
        self.ps.psi = cp.load("charged_arr_repo/random_psi.npy")
        self.integrator.ETD()
        self.ps.get_densities()
        cp.testing.assert_allclose(cp.load("charged_arr_repo/integrated_phi_all.npy"), self.ps.phi_all, rtol=1e-13)

class NeutralNumericTests(unittest.TestCase):
    def setUp(self):
        self.integrator, self.ps = build_polysystem(0.0)

    def test_get_density(self):
        #tests that the density for a random array from file of chemical potential
        # gives the correct matching density
        self.ps.w_all = cp.load("neutral_arr_repo/random_w_all.npy")
        self.ps.get_densities()
        cp.testing.assert_allclose(cp.load("neutral_arr_repo/phi_all.npy"), self.ps.phi_all, rtol=1e-13)


    def test_integrator(self):
        #checks that one step of integration gives the same results as previous after
        # random number has been reset
        cp.random.seed(0)
        self.ps.w_all = cp.load("neutral_arr_repo/random_w_all.npy")
        self.integrator.ETD()
        self.ps.get_densities()
        cp.testing.assert_allclose(cp.load("neutral_arr_repo/integrated_phi_all.npy"), self.ps.phi_all, rtol=1e-13)

class UnitTests(unittest.TestCase):
    def setUp(self):
        self.A_mon = p.Monomer("A", 1.)
        self.B_mon = p.Monomer("B", -1.)
        self.monomers = [self.A_mon, self.B_mon]
        self.w_all = cp.load("function_tests/random_w_all.npy")
        self.P_dict = {self.A_mon : self.w_all[0], self.B_mon : self.w_all[1]}
        grid_spec = (128,128)
        box_length = (10,10)
        self.grid = p.Grid(box_length=box_length, grid_spec = grid_spec)
        self.q_r = self.q_r_dag = cp.ones_like(self.w_all[0])
        self.int_width = 1/30

    def test_one_step(self):
        cp.testing.assert_array_equal(cp.load("function_tests/one_step_A.npy"), s_step(self.q_r, 1/30, self.w_all[0], self.grid))
    
    def test_A_MDE(self):
        poly = p.Polymer("A", 1, [(self.A_mon, 1)])
        poly.build_working_polymer(self.int_width, poly.total_length / 1)
        q_r_hold, q_r_dag_hold = integrate_s(poly.struct, poly.h_struct, self.P_dict, self.q_r, self.q_r_dag, self.grid)
        cp.testing.assert_array_equal(cp.load("function_tests/A_q_r.npy"), q_r_hold) 
        cp.testing.assert_array_equal(cp.load("function_tests/A_q_r.npy"), q_r_hold) 
        
    def test_AB_MDE(self):
        poly = p.Polymer("AB", 1, [(self.A_mon, .5), (self.B_mon, .5)])
        poly.build_working_polymer(self.int_width, poly.total_length / 1)
        q_r_hold, q_r_dag_hold = integrate_s(poly.struct, poly.h_struct, self.P_dict, self.q_r, self.q_r_dag, self.grid)
        cp.testing.assert_array_equal(cp.load("function_tests/AB_q_r.npy"), q_r_hold) 
        cp.testing.assert_array_equal(cp.load("function_tests/AB_q_r.npy"), q_r_hold) 
        
    def test_ABA_MDE(self):
        poly = p.Polymer("ABA", 1, [(self.A_mon, .25), (self.B_mon, .5), (self.A_mon, .25)])
        poly.build_working_polymer(self.int_width, poly.total_length / 1)
        q_r_hold, q_r_dag_hold = integrate_s(poly.struct, poly.h_struct, self.P_dict, self.q_r, self.q_r_dag, self.grid)
        cp.testing.assert_array_equal(cp.load("function_tests/ABA_q_r.npy"), q_r_hold) 
        cp.testing.assert_array_equal(cp.load("function_tests/ABA_q_r.npy"), q_r_hold) 
        
    def test_ABAB_MDE(self):
        poly = p.Polymer("ABA", 1, [(self.A_mon, .1), (self.B_mon, .2), (self.A_mon, .3), (self.B_mon, .4)])
        poly.build_working_polymer(self.int_width, poly.total_length / 1)
        q_r_hold, q_r_dag_hold = integrate_s(poly.struct, poly.h_struct, self.P_dict, self.q_r, self.q_r_dag, self.grid)
        cp.testing.assert_array_equal(cp.load("function_tests/ABAB_q_r.npy"), q_r_hold) 
        cp.testing.assert_array_equal(cp.load("function_tests/ABAB_q_r.npy"), q_r_hold) 
        
        

if __name__ == '__main__':
    # Create test suites
    charged_suite = unittest.TestLoader().loadTestsFromTestCase(ChargedNumericTests)
    neutral_suite = unittest.TestLoader().loadTestsFromTestCase(NeutralNumericTests)
    unit_suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)

    # Create a test runner
    runner = unittest.TextTestRunner()

    # Run the test suites
    runner.run(unittest.TestSuite([charged_suite, neutral_suite, unit_suite]))

