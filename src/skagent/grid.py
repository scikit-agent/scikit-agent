"""
Tools for building state and shock space grids.
"""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Grid:
    """
    Parameters:
    config (dict): A dictionary where each key represents a variable, and the value is another
        dictionary with the following keys:
        - "min" (float): The minimum value for the variable.
        - "max" (float): The maximum value for the variable.
        - "count" (int): The number of points to generate for the variable.

    """

    def __init__(self, config, torch=True):
        self.labels = list(config.keys())
        self.values = make_grid(config)

        if torch:
            self.torch()

    def torch(self):
        self.values = torch.FloatTensor(self.values).to(device)
        return self

    def to_dict(self):
        """
        Returns a data structure, key: column,
        similar to tensordict or structured array.
        """
        return dict(zip(self.labels, self.values.T))

    def __getitem__(self, sym):
        # TODO: fix the dict creation step to improve performance
        return self.to_dict()[sym]


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

    # patching, this should be codified as a new type
    # all_g.labels = list(config.keys())

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


def torched(grid):
    tens = torch.FloatTensor(grid).to(device)

    # patching, this should be codified as a new type
    # tens.labels = grid.labels

    return tens
