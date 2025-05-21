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
    Array of shape (len(arrays), len(arrays[0]) * len(arrays[1]) * ... * len(arrays[-1]))
    where each row contains the Cartesian product.
    """
    # Create meshgrid
    meshes = np.meshgrid(*arrays, indexing="ij")

    # Stack and reshape
    cartesian = np.stack(meshes, axis=-1)

    # Reshape to get the desired output
    return cartesian.reshape(-1, len(arrays))
