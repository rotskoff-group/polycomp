import copy
import cupy as cp


def clip(array, minimum, maximum):
    """
    Function to clip an array between a minimum and maximum value.

    Parameters
    ----------
    array : numpy.ndarray
        Array to be clipped
    minimum : float
        Minimum value to clip array to
    maximum : float
        Maximum value to clip array to
    """
    if minimum >= maximum:
        raise ValueError("Minium must be less than maximum")
    clipped_array = copy.copy(array)
    clipped_array[array < minimum] = minimum
    clipped_array[array > maximum] = maximum
    return clipped_array
