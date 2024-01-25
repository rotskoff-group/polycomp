import polycomp.grid as grid
import matplotlib.pyplot as plt
import cupy as cp
import math


def draw_circle(center, radius, grid):
    # Probably one of the single most over-engineered pieces of code I've every
    # written, this just takes a circle with a given center and radius and calculates
    # the corresponding density on some grid

    # we are going to want the area and the arc length, but we will collect the
    # chord length for now
    area = cp.zeros_like(grid.k2)
    chord = cp.zeros_like(grid.k2)
    ind = cp.zeros_like(grid.k2, dtype=int)

    # center position and radius
    pos = center
    rad = radius

    # We want to get the signed distances from the center of the circle to all the
    # grid points, we can ignore periodic boundary conditions because the circle is
    # centered
    disp = cp.sum(((grid.grid.T - pos).T) ** 2, axis=0) ** (0.5)
    dist = (((grid.grid.T - pos.T)) % (grid.l)).T
    sign = cp.sign((dist.T % grid.l.T) - grid.l.T / 2).T
    sign[sign == 0] = 1
    dist = cp.abs(dist.T - (grid.l) * (dist.T > (grid.l / 2))).T * sign
    # we want to get an indicator of how many corners of each grid point are in the
    # circle
    for edge in ([1, 1], [-1, 1], [1, -1], [-1, -1]):
        edge_dist = cp.abs(dist.T + (grid.dl / 2 * cp.array(edge))).T
        disp = cp.sum(edge_dist**2, axis=0) ** (0.5)
        ind[disp <= rad] += 1
    # Anything with all 4 corners enclosed is fully enclosed
    area[ind == 4] = grid.dV

    # These are the lines that all of the x and y gridlines lie on
    x_vals = dist[0, 0]
    x_lines = cp.append((x_vals + grid.dl[0] / 2), (x_vals[-1] - grid.dl[0] / 2))
    y_vals = dist[1, :, 0]
    y_lines = cp.append((y_vals + grid.dl[1] / 2), (y_vals[-1] - grid.dl[1] / 2))

    # These are the unsigned intercept distances to each gridline
    y_ints = cp.sqrt(rad**2 - x_lines**2)
    x_ints = cp.sqrt(rad**2 - y_lines**2)

    # We are going to try to assign each gridpoint four intercepts one for each side
    # of the grid cell
    ints = cp.zeros((*grid.k2.shape, 4, 2))
    ints_disp = cp.zeros((*grid.k2.shape, 4))
    ints[:, :, 0, 0] = x_lines[:-1]
    ints[:, :, 0, 1] = cp.abs(y_ints[:-1])
    ints[:, :, 1, 0] = x_lines[1:]
    ints[:, :, 1, 1] = cp.abs(y_ints[1:])
    ints = cp.swapaxes(ints, 0, 1)
    ints[:, :, 2, 1] = y_lines[:-1]
    ints[:, :, 2, 0] = x_ints[:-1]
    ints[:, :, 3, 1] = y_lines[1:]
    ints[:, :, 3, 0] = x_ints[1:]
    ints = cp.swapaxes(ints, 0, 1)

    # We want to replace every intercept that is not within the gridlines with nan
    x_ints = ints[:, :, :, 0]
    y_ints = ints[:, :, :, 1]
    x_nan_finder = cp.logical_or(
        cp.abs(x_ints[:, :, 2:4]).T > cp.amax(cp.abs(x_ints[:, :, 0:2]), axis=-1).T,
        cp.abs(x_ints[:, :, 2:4]).T < cp.amin(cp.abs(x_ints[:, :, 0:2]), axis=-1).T,
    ).T
    y_nan_finder = cp.logical_or(
        cp.abs(y_ints[:, :, 0:2]).T > cp.amax(cp.abs(y_ints[:, :, 2:4]), axis=-1).T,
        cp.abs(y_ints[:, :, 0:2]).T < cp.amin(cp.abs(y_ints[:, :, 2:4]), axis=-1).T,
    ).T
    x_ints[:, :, 2:4][x_nan_finder] = float("nan")
    y_ints[:, :, 0:2][y_nan_finder] = float("nan")
    # At this point any intercept that is not within the grid lines is the negative
    # of the correct value, so we are going to flip the sign on all of those
    bad_x = cp.logical_or(
        x_ints[:, :, 2:4].T < cp.amin(x_ints[:, :, 0:2], axis=-1).T,
        x_ints[:, :, 2:4].T > cp.amax(x_ints[:, :, 0:2], axis=-1).T,
    ).T
    bad_y = cp.logical_or(
        y_ints[:, :, 0:2].T < cp.amin(y_ints[:, :, 2:4], axis=-1).T,
        y_ints[:, :, 0:2].T > cp.amax(y_ints[:, :, 2:4], axis=-1).T,
    ).T
    x_ints[:, :, 2:4] = x_ints[:, :, 2:4] * (1 - 2 * bad_x)
    y_ints[:, :, 0:2] = y_ints[:, :, 0:2] * (1 - 2 * bad_y)
    ints[:, :, :, 0] = x_ints
    ints[:, :, :, 1] = y_ints

    # With all the intercepts defined, we can now solve the case with one corner in
    # the circle

    case_1 = ints[ind == 1]
    area_1 = area[ind == 1]
    chord_1 = chord[ind == 1]

    use_int = cp.zeros((2, 2))
    # the basic plan is the find the area of the triangle in the corner, add that to
    # the area, then record the hypotenuse length for later as the chord length
    for i in range(case_1.shape[0]):
        use_int[0] = case_1[i, 2 + cp.nanargmin(cp.abs(case_1[i, 2:4, 0]))]
        use_int[1] = case_1[i, cp.nanargmin(cp.abs(case_1[i, 0:2, 1]))]
        change = cp.abs(use_int[0] - use_int[1])
        change = change % grid.dl
        area_1[i] = change[0] * change[1] / 2
        chord_1[i] = cp.sqrt(cp.sum(change**2))
    area[ind == 1] = area_1
    chord[ind == 1] = chord_1

    # Solve the case with two corners in the circle
    case_2 = ints[ind == 2]
    area_2 = area[ind == 2]
    chord_2 = chord[ind == 2]

    get_nan = cp.sum(cp.isnan(case_2), axis=1)

    use_ints = cp.zeros((case_2.shape[0], 2, 2))
    # General plan, determine whether the cell is to the side of the circle of over/
    # under. Then find the two intercepts and the bottom and use those to get the
    # area within the cell and the chord length
    for i in range(case_2.shape[0]):
        if get_nan[i, 1] == 0:
            use_int = case_2[i, 0:2]
            bottom = case_2[i, 2:4, 1][cp.argmin(cp.abs(case_2[i, 2:4, 1]))]
            area_2[i] = (
                cp.abs(
                    (use_int[0, 0] - use_int[1, 0])
                    * (use_int[1, 1] + use_int[0, 1] - 2 * bottom)
                )
                / 2
            )
            chord_2[i] = cp.sum((use_int[0] - use_int[1]) ** 2) ** (0.5)
        else:
            use_int = case_2[i, 2:4]
            bottom = case_2[i, 0:2, 0][cp.argmin(cp.abs(case_2[i, 0:2, 0]))]
            area_2[i] = (
                cp.abs(
                    (use_int[0, 1] - use_int[1, 1])
                    * (use_int[1, 0] + use_int[0, 0] - 2 * bottom)
                )
                / 2
            )
            chord_2[i] = cp.sum((use_int[0] - use_int[1]) ** 2) ** (0.5)
    area[ind == 2] = area_2
    chord[ind == 2] = chord_2

    # Solve case with 3 corners

    case_3 = ints[ind == 3]
    area_3 = area[ind == 3]
    chord_3 = chord[ind == 3]

    use_int = cp.zeros((2, 2))
    # Essentially the same as the 1 case, except we will subtract off the triangle
    # from the area of the whole cell
    for i in range(case_3.shape[0]):
        use_int[0] = case_3[i, 2 + cp.nanargmax(cp.abs(case_3[i, 2:4, 0]))]
        use_int[1] = case_3[i, cp.nanargmax(cp.abs(case_3[i, 0:2, 1]))]
        # if you exactly hit the corner you can have repeated indices, this fixes
        # that case
        if cp.all(use_int[0] == use_int[1]):
            hold = case_3[i][case_3[i] != use_int[0]]
            if cp.any(cp.isnan(hold[0:1])):
                use_int[1] = hold[2:4]
            else:
                use_int[1] = hold[0:2]
        change = cp.abs(use_int[0] - use_int[1])
        change = change % grid.dl
        change[cp.isclose(change, 0)] = grid.dl[cp.isclose(change, 0)]
        area_3[i] = grid.dV - change[0] * change[1] / 2
        chord_3[i] = cp.sqrt(cp.sum(change**2))
    area[ind == 3] = area_3
    chord[ind == 3] = chord_3

    # Use some simple trig to calculate the arc length
    arc = rad * 2 * cp.arcsin(chord / (2 * rad))
    theta = 2 * cp.arcsin(chord / (2 * rad))

    # Add the segment area corresponding to each chord
    area += (1 / 2) * (theta - cp.sin(theta)) * rad**2

    # Check against an analytical formula
    # print(cp.sum(area / (math.pi * rad**2)))
    # print(cp.sum(arc) / (math.pi * 2 * rad))
    return area, chord
