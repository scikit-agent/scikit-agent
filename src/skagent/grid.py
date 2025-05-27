"""
Tools for building state and shock space grids.
"""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torched(grid):
    tens = torch.FloatTensor(grid).to(device)
    return tens


def make_grid(config):
    """
    Make a 'grid' of values based on the provided configuration.

    Parameters:
    config (dict): A dictionary where each key represents a variable, and the value is another
        dictionary with the following keys:
        - "min" (float): The minimum value for the variable.
        - "max" (float): The maximum value for the variable.
        - "count" (int): The number of points to generate for the variable.

    Returns:
    numpy.ndarray: A NumPy array of shape `(product_of_counts, num_variables)`, where
        `product_of_counts` is the product of all `count` values in the `config` dictionary,
        and `num_variables` is the number of keys in the `config`.
    """
    arrays = []

    for sym in config:
        g = np.linspace(config[sym]["min"], config[sym]["max"], config[sym]["count"])
        arrays.append(g)

    all_g = cartesian_product(*arrays)

    return all_g


def cartesian_product(*arrays):
    """
    Create a Cartesian product of input arrays.

    Parameters:
    *arrays: Variable length arrays to compute product

    Returns:
    Array of shape (product_of_lengths, num_arrays)
    where `product_of_lengths` is the product of the lengths of the input arrays,
    and `num_arrays` is the number of input arrays. Each row contains one element
    of the Cartesian product.
    """
    # Create meshgrid
    meshes = np.meshgrid(*arrays, indexing="ij")

    # Stack and reshape
    cartesian = np.stack(meshes, axis=-1)

    # Reshape to get the desired output
    return cartesian.reshape(-1, len(arrays))
