import cupy as cp


class Grid(object):
    """
    Grid object for a polymer simulation.

    Attributes:
        grid_spec (tuple):
            Number of grid points along each axis.
        ndims (int):
            Dimension of the system.
        l (cparray):
            CPArray for the length of the box along each axis.
        dl (cparray):
            CPArray for the length of the box along each axis for the unit cell.
        V (float):
            Total box volume.
        dV (float):
            Volume of the unit cell.
        grid (cparray):
            Float array of the (x, ...) position at each grid point.
        kgrid (cparray):
            Complex grid of (x, ...) k Fourier-transformed positions at each k point.
        k1 (cparray):
            Complex grid of (x, ...) L1 norm distances at each k point.
        k2 (cparray):
            Complex grid of (x, ...) L2 norm distances at each k point.
    """

    def __init__(self, box_length, grid_spec):
        """
        Initialize Grid

        Builds grid object for given input values

        Parameters:
            box_length (tuple): tuple of floats representing the length of each axis of box
            grid_spec (tuple): tuple of ints representing the number of grid points along each axis

        Raises:
            
            ValueError:
                Raises error if the box length is not a tuple
        """
        super(Grid, self).__init__()

        self.grid_spec = grid_spec
        self.ndims = len(self.grid_spec)
        if type(box_length) is tuple:
            self.l = cp.array(box_length)
        else:
            raise ValueError("box_length is not tuple")
        self.update_l(self.l)

    def update_l(self, new_l):
        """
        Set up everything inside the grid.

        Builds various useful structures that provide information about the
        grid, including Fourier-transformed positions and real position arrays.

        Parameters:
            new_l (tuple of float): Tuple of floats representing the new box lengths.

        Raises:

            ValueError: 
                Raises error if the box length is not a tuple
        """

        self.l = cp.array(new_l)

        # Total volume
        self.V = cp.prod(cp.array(self.l))

        # Grid of real positions
        self.grid = cp.asarray(
            cp.meshgrid(
                *[
                    cp.linspace(0, l, n)
                    for n, l in zip(
                        self.grid_spec,
                        self.l * (1 - 1 / cp.array(self.grid_spec)),
                    )
                ]
            )
        )

        # Grid of k positions
        self.kgrid = cp.asarray(
            cp.meshgrid(
                *[
                    2
                    * cp.pi
                    / l
                    * cp.concatenate(
                        (cp.arange(0, n / 2 + 1), cp.arange(-n / 2 + 1, 0)),
                        axis=None,
                    )
                    for n, l in zip(self.grid_spec, self.l)
                ]
            )
        )

        self.k1 = cp.sum(self.kgrid, axis=0)
        self.k2 = cp.sum(self.kgrid**2, axis=0)
        self.dV = self.V / self.k2.size
        self.dl = self.l / cp.array(self.grid_spec)

        return
