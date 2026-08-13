from typing import Tuple, Union

import cupy as cp

ArrayLike = Union[Tuple[float, ...], list, cp.ndarray]


class Grid:
    """
    This class manages the grids (real and k-space) needed for the field theoretic
    simulations. It stores and pre-computes useful quantities for the
    modified diffusion equation and integrators to use elsewhere.

    Parameters
    ----------

    box_length
        Lengths of the simulation box along each axis $L_i$.
    grid_spec
        Number of lattice points along each axis $M_{Li}$.

    Attributes
    ----------

    grid_spec : Tuple[int, ...]
        Number of lattice points along each axis $M_{Li}$.
    ndims : int
        Dimension of the system $d$.
    l : cp.ndarray of float
        Lengths of the simulation box along each axis $L_i$.
    dl : cp.ndarray of float
        Lengths of the unit cell along each axis $\\Delta L_i$.
    V : float
        Total box volume $V$.
    dV : float
        Volume of the unit cell $dV$.
    grid : cp.ndarray of float
        Real-space coordinates of each grid point.
        Shape is ($d$, $N_x$, $N_y$, ...). grid[0] is a grid of all
        x-coordinates, same for other dimensions
    kgrid : cp.ndarray of float
        Grid of (x, ...) k Fourier-transformed positions at each k point.
    k1 : cp.ndarray of float
        Grid of (x, ...) L1 norm distances at each k point.
    k2 : cp.ndarray of float
        Grid of (x, ...) L2 norm distances at each k point.

    Raises
    ------

    ValueError
        Raises error if the box length is not a tuple
    """

    def __init__(
        self, box_length: Tuple[float, ...], grid_spec: Tuple[int, ...]
    ) -> None:
        super().__init__()

        self.grid_spec = grid_spec
        self.ndims = len(self.grid_spec)
        if isinstance(box_length, tuple):
            self.l = cp.array(box_length)
        else:
            raise ValueError("box_length is not tuple")
        self.update_l(self.l)

    def update_l(self, new_l: ArrayLike) -> None:
        """
        This function reconstructs the grid, using the previous gridding but for a new
        box size. Performs required operations for all dependent parameters to be
        correctly set for new grid.

        Parameters
        ----------

        new_l
            New box lengths $L_i$ to be assigned

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
                    2 * cp.pi * cp.fft.fftfreq(n, d=l/n)
                    for n, l in zip(self.grid_spec, self.l)
                ]
            )
        )

        self.k1 = cp.sum(self.kgrid, axis=0)
        self.k2 = cp.sum(self.kgrid**2, axis=0)
        self.dV = self.V / self.k2.size
        self.dl = self.l / cp.array(self.grid_spec)

        return
