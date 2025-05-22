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
    Make a 'grid' of values. ...
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
