import unittest

import cupy as cp

import polycomp.ft_system as p
from polycomp.mde import integrate_s, s_step


def build_polysystem(charge):
    cp.random.seed(0)

    A_mon = p.Monomer("A", charge)
    B_mon = p.Monomer("B", -charge)
    S_mon = p.Monomer("S", 0)
    monomers = [A_mon, B_mon, S_mon]

    FH_terms = {
        frozenset({A_mon}): 2,
        frozenset({B_mon}): 2,
        frozenset({S_mon}): 2,
        frozenset({A_mon, B_mon}): 3,
        frozenset({A_mon, S_mon}): 3,
        frozenset({B_mon, S_mon}): 3,
    }

    N = 5

    A_poly = p.Polymer("A", N, [(A_mon, 1)])
    B_poly = p.Polymer("B", N, [(B_mon, 1)])
    polymers = [A_poly, B_poly]
    spec_dict = {A_poly: 0.3, B_poly: 0.3, S_mon: 1.5 * N}

    grid_spec = (128, 128)
    box_length = (10, 10)
    grid = p.Grid(box_length=box_length, grid_spec=grid_spec)

    smear = 0.1
    ps = p.PolymerSystem(
        monomers,
        polymers,
        spec_dict,
        FH_terms,
        grid,
        smear,
        salt_conc=0.0 * N,
        integration_width=1 / 30,
    )

    relax_rates = cp.array([1 * ps.grid.dV] * (ps.w_all.shape[0]))
    w_temps = cp.array([0.003 + 0j] * (ps.w_all.shape[0]))
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
        # tests that the density for a random array from file of chemical potential
        # gives the correct matching density
        self.ps.w_all = cp.load("charged_arr_repo/random_w_all.npy")
        self.ps.psi = cp.load("charged_arr_repo/random_psi.npy")
        self.ps.get_densities()
        cp.testing.assert_allclose(
            cp.load("charged_arr_repo/phi_all.npy"), self.ps.phi_all, rtol=1e-13
        )

    def test_integrator(self):
        # checks that one step of integration gives the same results as previous after
        # random number has been reset
        cp.random.seed(0)
        self.ps.w_all = cp.load("charged_arr_repo/random_w_all.npy")
        self.ps.psi = cp.load("charged_arr_repo/random_psi.npy")
        self.integrator.ETD()
        self.ps.get_densities()
        cp.testing.assert_allclose(
            cp.load("charged_arr_repo/integrated_phi_all.npy"),
            self.ps.phi_all,
            rtol=1e-6,
            atol=1e-8,
        )


class NeutralNumericTests(unittest.TestCase):
    def setUp(self):
        self.integrator, self.ps = build_polysystem(0.0)

    def test_get_density(self):
        # tests that the density for a random array from file of chemical potential
        # gives the correct matching density
        self.ps.w_all = cp.load("neutral_arr_repo/random_w_all.npy")
        self.ps.get_densities()
        cp.testing.assert_allclose(
            cp.load("neutral_arr_repo/phi_all.npy"), self.ps.phi_all, rtol=1e-13
        )

    def test_integrator(self):
        # checks that one step of integration gives the same results as previous after
        # random number has been reset
        cp.random.seed(0)
        self.ps.w_all = cp.load("neutral_arr_repo/random_w_all.npy")
        self.integrator.ETD()
        self.ps.get_densities()
        cp.testing.assert_allclose(
            cp.load("neutral_arr_repo/integrated_phi_all.npy"),
            self.ps.phi_all,
            rtol=1e-13,
        )


class UnitTests(unittest.TestCase):
    def setUp(self):
        self.A_mon = p.Monomer("A", 1.0)
        self.B_mon = p.Monomer("B", -1.0)
        self.monomers = [self.A_mon, self.B_mon]
        self.w_all = cp.load("function_tests/random_w_all.npy")
        self.P_dict = {self.A_mon: self.w_all[0], self.B_mon: self.w_all[1]}
        grid_spec = (128, 128)
        box_length = (10, 10)
        self.grid = p.Grid(box_length=box_length, grid_spec=grid_spec)
        self.q_r = self.q_r_dag = cp.ones_like(self.w_all[0])
        self.int_width = 1 / 30

    def test_one_step(self):
        cp.testing.assert_allclose(
            cp.load("function_tests/one_step_A.npy"),
            s_step(self.q_r, 1 / 30, self.w_all[0], self.grid),
            rtol=1e-13
        )

    def test_A_MDE(self):
        poly = p.Polymer("A", 1, [(self.A_mon, 1)])
        poly.build_working_polymer(self.int_width, poly.total_length / 1)
        q_r_hold, q_r_dag_hold = integrate_s(
            poly.struct, poly.h_struct, self.P_dict, self.q_r, self.q_r_dag, self.grid
        )
        cp.testing.assert_allclose(cp.load("function_tests/A_q_r.npy"), q_r_hold, rtol=1e-13)
        cp.testing.assert_allclose(cp.load("function_tests/A_q_r.npy"), q_r_hold, rtol=1e-13)

    def test_AB_MDE(self):
        poly = p.Polymer("AB", 1, [(self.A_mon, 0.5), (self.B_mon, 0.5)])
        poly.build_working_polymer(self.int_width, poly.total_length / 1)
        q_r_hold, q_r_dag_hold = integrate_s(
            poly.struct, poly.h_struct, self.P_dict, self.q_r, self.q_r_dag, self.grid
        )
        cp.testing.assert_allclose(cp.load("function_tests/AB_q_r.npy"), q_r_hold, rtol=1e-13)
        cp.testing.assert_allclose(cp.load("function_tests/AB_q_r.npy"), q_r_hold, rtol=1e-13)

    def test_ABA_MDE(self):
        poly = p.Polymer(
            "ABA", 1, [(self.A_mon, 0.25), (self.B_mon, 0.5), (self.A_mon, 0.25)]
        )
        poly.build_working_polymer(self.int_width, poly.total_length / 1)
        q_r_hold, q_r_dag_hold = integrate_s(
            poly.struct, poly.h_struct, self.P_dict, self.q_r, self.q_r_dag, self.grid
        )
        cp.testing.assert_allclose(cp.load("function_tests/ABA_q_r.npy"), q_r_hold, rtol=1e-13)
        cp.testing.assert_allclose(cp.load("function_tests/ABA_q_r.npy"), q_r_hold, rtol=1e-13)

    def test_ABAB_MDE(self):
        poly = p.Polymer(
            "ABA",
            1,
            [
                (self.A_mon, 0.1),
                (self.B_mon, 0.2),
                (self.A_mon, 0.3),
                (self.B_mon, 0.4),
            ],
        )
        poly.build_working_polymer(self.int_width, poly.total_length / 1)
        q_r_hold, q_r_dag_hold = integrate_s(
            poly.struct, poly.h_struct, self.P_dict, self.q_r, self.q_r_dag, self.grid
        )
        cp.testing.assert_allclose(cp.load("function_tests/ABAB_q_r.npy"), q_r_hold, rtol=1e-13)
        cp.testing.assert_allclose(cp.load("function_tests/ABAB_q_r.npy"), q_r_hold, rtol=1e-13)


class PolymerBuildTests(unittest.TestCase):
    """
    Tests the logic of Polymer.build_working_polymer to ensure correct
    discretization of the polymer chain.
    """

    def setUp(self):
        """Set up monomer objects for testing discretization."""
        self.A_mon = p.Monomer("A", 1.0)
        self.B_mon = p.Monomer("B", -1.0)

    def test_build_homopolymer_perfect_division(self):
        """
        Tests discretization when a block length is a perfect multiple of h.
        """
        poly = p.Polymer("A", 1.0, [(self.A_mon, 1.0)])
        h, total_h = 0.1, 1.0
        poly.build_working_polymer(h, total_h)

        # Expected: 1.0 / 0.1 = 10 segments, each of length 0.1
        self.assertEqual(len(poly.h_struct), 10)
        cp.testing.assert_allclose(poly.h_struct, cp.full(10, 0.1))
        self.assertTrue(all(mon == self.A_mon for mon in poly.struct))
        self.assertAlmostEqual(cp.sum(poly.h_struct).item(), total_h)

    def test_build_homopolymer_imperfect_division(self):
        """
        Tests discretization when a block length is not a multiple of h,
        checking the ceiling division and segment length averaging.
        """
        poly = p.Polymer("A", 1.0, [(self.A_mon, 1.0)])
        h, total_h = 0.3, 1.0
        poly.build_working_polymer(h, total_h)

        # Expected: ceil(1.0 / 0.3) = 4 segments, each of length 1.0 / 4 = 0.25.
        self.assertEqual(len(poly.h_struct), 4)
        cp.testing.assert_allclose(poly.h_struct, cp.full(4, 0.25))
        self.assertTrue(cp.all(poly.h_struct <= h))
        self.assertAlmostEqual(cp.sum(poly.h_struct).item(), total_h)

    def test_build_diblock_polymer(self):
        """
        Tests that a multi-block polymer is discretized correctly, respecting
        the sequence and length of each block.
        """
        poly = p.Polymer("AB", 2.0, [(self.A_mon, 0.5), (self.B_mon, 0.5)])
        h, total_h = 0.4, 2.0
        poly.build_working_polymer(h, total_h)

        # Block A (length 1.0): ceil(1.0 / 0.4) = 3 segments of 1.0/3 each.
        # Block B (length 1.0): ceil(1.0 / 0.4) = 3 segments of 1.0/3 each.
        self.assertEqual(len(poly.h_struct), 6)
        self.assertTrue(all(mon == self.A_mon for mon in poly.struct[:3]))
        self.assertTrue(all(mon == self.B_mon for mon in poly.struct[3:]))
        expected_lengths = cp.array([1.0 / 3] * 6)
        cp.testing.assert_allclose(poly.h_struct, expected_lengths)
        self.assertAlmostEqual(cp.sum(poly.h_struct).item(), total_h)

    def test_build_multiblock_has_even_segments_per_block(self):
        """
        Tests the core assumption that for any given block, all of its
        discretized segments are of equal length.
        """
        poly = p.Polymer(
            "ABA",
            2.1,
            [(self.A_mon, 0.7 / 2.1), (self.B_mon, 0.9 / 2.1), (self.A_mon, 0.5 / 2.1)],
        )
        h, total_h = 0.3, 2.1
        poly.build_working_polymer(h, total_h)

        # Expected segments: Block1=3, Block2=3, Block3=2. Total=8
        self.assertEqual(len(poly.h_struct), 8)
        h_block1, h_block2, h_block3 = (
            poly.h_struct[0:3],
            poly.h_struct[3:6],
            poly.h_struct[6:8],
        )

        # Assert that segments WITHIN each block are uniform
        self.assertEqual(
            len(cp.unique(h_block1)), 1, "Segments in block 1 are not uniform"
        )
        cp.testing.assert_allclose(h_block1[0], 0.7 / 3)
        self.assertEqual(
            len(cp.unique(h_block2)), 1, "Segments in block 2 are not uniform"
        )
        cp.testing.assert_allclose(h_block2[0], 0.9 / 3)
        self.assertEqual(
            len(cp.unique(h_block3)), 1, "Segments in block 3 are not uniform"
        )
        cp.testing.assert_allclose(h_block3[0], 0.5 / 2)
        self.assertAlmostEqual(cp.sum(poly.h_struct).item(), total_h)

    def test_build_polymer_called_twice_raises_error(self):
        """
        Tests the guard clause that prevents a polymer from being built twice.
        """
        poly = p.Polymer("A", 1.0, [(self.A_mon, 1.0)])
        poly.build_working_polymer(h=0.1, total_h=1.0)
        with self.assertRaisesRegex(
            ValueError, "polymer structure should only be built once"
        ):
            poly.build_working_polymer(h=0.1, total_h=1.0)


class PropertyTests(unittest.TestCase):
    """
    Tests fundamental mathematical and physical properties of the simulation
    algorithms. These tests are self-contained and use smooth fields to
    ensure they test the algorithm under its designed operating conditions.
    """

    def setUp(self):
        self.integrator, self.ps = build_polysystem(0.0)

        # --- Create SMOOTH fields for all tests ---
        x = self.ps.grid.grid[0]
        y = self.ps.grid.grid[1]
        Lx = self.ps.grid.l[0]
        Ly = self.ps.grid.l[1]

        # A smooth initial condition for the propagator `q`
        self.q_initial_smooth = cp.sin(2 * cp.pi * x / Lx) + 1j * cp.cos(
            2 * cp.pi * y / Ly
        )

        # A smooth potential field `w_P`
        self.w_P_smooth = (
            0.1 * cp.cos(2 * cp.pi * x / Lx) * cp.sin(2 * cp.pi * y / Ly)
        ).astype(cp.complex128)

    def test_mass_is_conserved_during_etd(self):
        """
        Tests that a single integrator step conserves the total mass for each species.
        """
        self.ps.w_all = cp.load("neutral_arr_repo/random_w_all.npy")
        self.ps.get_densities()
        initial_masses = cp.array([cp.sum(phi) for phi in self.ps.phi_all])
        self.integrator.ETD()
        final_masses = cp.array([cp.sum(phi) for phi in self.ps.phi_all])
        cp.testing.assert_allclose(initial_masses, final_masses, rtol=1e-13)

    def test_normal_mode_transform_is_reversible(self):
        """
        Tests that the normal mode transformation is reversible within float64
        precision.
        """
        w_density_space = cp.random.rand(*self.ps.w_all.shape) + 1j * cp.random.rand(
            *self.ps.w_all.shape
        )
        w_normal_space = self.ps.map_norm_from_dens(w_density_space)
        w_reconstructed = self.ps.map_dens_from_norm(w_normal_space)
        cp.testing.assert_allclose(w_density_space, w_reconstructed, rtol=1e-13)

    def test_s_step_semigroup_property_for_diffusion(self):
        """
        Tests that the diffusion operator follows the semigroup property:
        prop(h1) * prop(h2) = prop(h1+h2).
        """
        w_zero = cp.zeros_like(self.q_initial_smooth, dtype=cp.complex128)
        h1 = self.ps.integration_width / 3.0
        h2 = self.ps.integration_width / 2.0
        q_two_steps = s_step(
            s_step(self.q_initial_smooth, h1, w_zero, self.ps.grid),
            h2,
            w_zero,
            self.ps.grid,
        )
        q_one_step = s_step(self.q_initial_smooth, h1 + h2, w_zero, self.ps.grid)

        # CORRECTED: Added a small absolute tolerance to handle floating point
        # noise on values that are mathematically zero.
        cp.testing.assert_allclose(q_one_step, q_two_steps, rtol=1e-13, atol=1e-14)

    def test_s_step_has_second_order_accuracy(self):
        """
        Confirms that `s_step` is second-order accurate by checking that its
        local error scales as O(h^3). This is the precondition for the RQM4
        method to be fourth-order accurate. This test now uses smooth fields.
        """
        h = self.ps.integration_width / 4.0  # Use a small h for better asymptotics

        # Calculate local error for step size h
        q1 = s_step(self.q_initial_smooth, h, self.w_P_smooth, self.ps.grid)
        q2 = s_step(
            s_step(self.q_initial_smooth, h / 2.0, self.w_P_smooth, self.ps.grid),
            h / 2.0,
            self.w_P_smooth,
            self.ps.grid,
        )
        error_h = cp.linalg.norm(q1 - q2)

        # Calculate local error for step size h/2
        q1_half = s_step(self.q_initial_smooth, h / 2.0, self.w_P_smooth, self.ps.grid)
        q2_half = s_step(
            s_step(self.q_initial_smooth, h / 4.0, self.w_P_smooth, self.ps.grid),
            h / 4.0,
            self.w_P_smooth,
            self.ps.grid,
        )
        error_h_half = cp.linalg.norm(q1_half - q2_half)

        self.assertTrue(
            error_h_half.item() > 1e-15,
            "Error is too small to measure ratio accurately.",
        )

        error_ratio = error_h / error_h_half

        # Assert that the error ratio is close to 8 (2^3), confirming O(h^3) local error
        self.assertGreater(
            error_ratio,
            7.5,
            f"Error ratio is {error_ratio:.2f}, should be ~8. s_step is "
            "NOT second-order accurate.",
        )
        self.assertLess(
            error_ratio,
            8.5,
            f"Error ratio is {error_ratio:.2f}, should be ~8. Scaling is unexpected.",
        )


if __name__ == "__main__":
    # Create test suites from all test classes
    charged_suite = unittest.TestLoader().loadTestsFromTestCase(ChargedNumericTests)
    neutral_suite = unittest.TestLoader().loadTestsFromTestCase(NeutralNumericTests)
    unit_suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)

    # Import the new test classes
    build_suite = unittest.TestLoader().loadTestsFromTestCase(PolymerBuildTests)
    property_suite = unittest.TestLoader().loadTestsFromTestCase(PropertyTests)

    # Create a test runner
    runner = unittest.TextTestRunner(verbosity=1)

    # Combine all suites into a single suite to run
    all_tests = unittest.TestSuite(
        [
            charged_suite,
            neutral_suite,
            unit_suite,
            build_suite,  # Add the new polymer build tests
            property_suite,  # Add the new property tests
        ]
    )

    # Run the test suites
    runner.run(all_tests)
