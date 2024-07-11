import cupy as cp
import numpy as np


class Monomer(object):
    """
    Class for the monomer of one species in the simulation.

    Attributes:
        name (string):
            Unique monomer name.
        has_volume (bool):
            Whether the monomer occupies volume.
        identity (string):
            Identity of the monomer, probably {polymer, solvent, salt}.
        charge (float):
            Charge of the monomer.
    """

    def __init__(self, name, charge, identity="solvent", has_volume=True):
        """
        Initialize Monomer object.

        Parameters:
            name (string):
                Unique monomer name.
            identity (string):
                What type of species the monomer is.
            has_volume (bool):
                Whether the monomer occupies volume.
        """

        self.name = name
        self.has_volume = has_volume
        self.identity = identity
        self.charge = charge

    def __repr__(self):
        return self.name


class Polymer(object):
    """
    Class to store all the information for one type of polymer.

    Attributes:
        name (string):
            Unique name of a polymer.
        total_length (float):
            Total length of the polymer for integration.
        block_structure (tuple):
            Tuple of dicts containing the length of component blocks within the
            polymer.
        struct (ndarray):
            Array of Monomer objects representing linear polymer structure.
        h_struct (cparray):
            Array of floats for the length of each section of the structure.
        fastener (cparray):
            cparray indicating where the polymer is fastened.
    """

    def __init__(self, name, total_length, block_structure, fastener=None):
        """
        Initialize Polymer object.

        Parameters:
            name (string):
                Unique name.
            total_length (float):
                Total length along the polymer.
            block_structure (tuple):
                Tuple of dictionaries mapping monomer objects to lengths along the polymer.
        """

        super(Polymer, self).__init__()
        self.name = name
        self.total_length = total_length
        self.block_structure = block_structure

        self.struct = None
        # Identify which species are in any polymer
        for monomer in set(p[0] for p in self.block_structure):
            if monomer.identity != "polymer":
                monomer.identity = "polymer"

        self.identity = "entire_polymer"
        self.fastener = fastener

    def __repr__(self):
        return str(self.block_structure)

    def build_working_polymer(self, h, total_h):
        """
        Build the polymer structure that will be used for integration.

        Built-in method to construct a working polymer during integration
        according to parameters specific to the simulation run.

        Parameters:
            h (float):
                Maximum integration segment length.

        Raises:
            ValueError:
                Raises an error if the polymer already has a built structure. At
                present, there is no reason that a polymer structure should be built
                more than once in a single simulation.
        """

        # Used to generate a string of h lengths and polymer species identities
        if self.struct is not None:
            raise ValueError("polymer structure should only be built once")
        hold_struct = []
        hold_h_struct = []
        where = 0.0
        end = 0.0

        # Splits up each block evenly while keeping h below target
        thresh = 1e-10
        for name_tuple in self.block_structure:
            end = name_tuple[1] * total_h
            units = int(end // h)
            if end % h > thresh:
                units += 1
            hold_struct = hold_struct + ([name_tuple[0]] * units)
            hold_h_struct = hold_h_struct + ([end / units] * units)

        self.struct = np.asarray(hold_struct)
        self.h_struct = cp.asarray(hold_h_struct, dtype="float64")

        #Also build a dictionary storing the total mass of each monomer
        mon_set = set([s[0] for s in self.block_structure])
        self.mon_mass = {}
        for mon in mon_set:
            self.mon_mass[mon] = float(cp.sum(self.h_struct[self.struct==mon]))
        return


class Brush(object):
    def __init__(self, name, density):
        self.name = name
        self.density = density / cp.average(density)

    def __repr__(self):
        return self.name


class Nanoparticle(object):
    def __init__(self, name, monomer_type, density):
        self.name = name
        self.density = density
        self.type = monomer_type

        self.type.identity = "Nanoparticle"

    def __repr__(self):
        return self.name

    def place_nps(self, positions):
        self.positions = positions
