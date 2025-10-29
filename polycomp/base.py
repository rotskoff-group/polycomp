import cupy as cp
import numpy as np


class Monomer(object):
    """
    Class for the monomer of one species in the simulation.

    Parameters
    ----------

    name
        Monomer identifier to be displayed publicly.
    charge
        Monomer charge.
    identity
        Monomer type (solvent or polymer).
    has_volume
        Whether the monomer occupies volume.

    Attributes
    ----------

    name : string
        Unique monomer name.
    charge : float
        Charge of the monomer.
    identity : string
        Identity of the monomer, usually {polymer, solvent, salt, Nanoparticle}.
    has_volume : bool
        Whether the monomer occupies volume.
    """

    def __init__(
        self,
        name: str,
        charge: float = 0,
        identity: str = "solvent",
        has_volume: bool = True,
    ) -> None:

        self.name = name
        self.has_volume = has_volume
        self.identity = identity
        self.charge = charge

    def __repr__(self):
        return self.name


class Polymer(object):
    """
    Class to store all the information for one type of polymer.

    Parameters
    ----------

    name
        Unique name.
    total_length
        Total length along the polymer. (Only accurate if sum of block lengths is 1)
    block_structure : array-like
        A list-like object of blocks defining the polymer architecture. Each
        block should be a list-like object of length 2, in the format
        `(Monomer, float_fractional_length)`.

        Example for a diblock: `[(A_mon, 0.5), (B_mon, 0.5)]`

    Attributes
    ----------

    name : string
        Unique name of a polymer.
    total_length : float
        Total length of the polymer for integration.
    block_structure : array-like
        A list-like object of blocks defining the polymer architecture. Format is
        `(Monomer, float_fractional_length)`.
    struct : np.ndarray
        Array of Monomer objects representing linear polymer structure.
    h_struct : cp.ndarray
        Array of floats for the length of each section of the structure.
    fastener : Brush
        Brush object indicating where the sequence is fastened. Fastening always occurs
        on the leading end.

    """

    def __init__(
        self, name: str, total_length: float, block_structure: tuple, fastener=None
    ) -> None:

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

    def build_working_polymer(self, h: float, total_h: float) -> None:
        """
        Build the polymer structure that will be used for integration.

        Built-in method to construct a working polymer during integration
        according to parameters specific to the simulation run. Normally called
        automatically as part of system setup.

        Parameters
        ----------

        h
            Maximum integration segment length $\\Delta s$.
        total_h
            Total length of the polymer $s_P$.

        Raises
        ------

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
        return


class Brush(object):
    """
    Class to store the location of a brushed surface. The brush density denotes the
    density of the affixed end of the polymer in space.

    Parameters
    ----------

    name
        Unique name.
    density
        Density in space (of some corresponding grid) of
        the fixed segment of the polymer.

    Attributes
    ----------

    name : str
        Unique name.
    density : cp.ndarray
        Spatial density of the attached brush end

    """

    def __init__(self, name: str, density: cp.ndarray) -> None:
        self.name = name
        self.density = density / cp.average(density)

    def __repr__(self):
        return self.name


class Nanoparticle(object):
    """

    Defines an inert nanoparticle with a fixed spatial density.

    The position of the nanoparticle is fixed unless manually changed.

    Parameters
    ----------

    name
        Unique name.
    monomer_type
        Monomer type associated with the nanoparticle in question (for FH interaction
        purposes).
    density
        Density in space (of some corresponding grid) of the nanoparticle.

    Attributes
    ----------

    name : str
        Unique name.
    density : cp.ndarray
        Spatial density of the nanoparticle.
    type : Monomer
        Monomer identity (for FH interaction purposes) of the nanoparticle.

    """

    def __init__(self, name: str, monomer_type: Monomer, density: cp.ndarray) -> None:
        self.name = name
        self.density = density
        self.type = monomer_type

        self.type.identity = "Nanoparticle"

    def __repr__(self):
        return self.name

    def place_nps(self, positions: cp.ndarray) -> None:
        """
        Updates the density of the nanoparticle position

        Parameters
        ----------

        positions
            Desired new density profile for the nanoparticles in the system.
        """

        self.density = positions
