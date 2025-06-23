"""
Tools for building neural networks for economic models.

This module provides neural network architectures specifically designed for solving
dynamic economic models, with particular emphasis on the MMW JME '21 framework
(Maliar, Maliar, and Winant, Journal of Monetary Economics, 2021).

Key Features
============

- **Configurable architectures**: Support for various layer counts, widths, and activation functions
- **MMW compatibility**: Specialized networks for MMW JME '21 reproduction with dual outputs
- **Economic model integration**: Seamless integration with skagent DBlock economic models
- **Constraint handling**: Built-in support for Fischer-Burmeister constraint penalties
- **Device management**: Automatic CUDA/CPU tensor handling

Neural Network Classes
======================

Basic Networks:
- Net: Simple feedforward network with configurable architecture
- FlexibleNet: Enhanced network with configurable layers, activations, and output transforms

Economic Model Networks:
- BlockPolicyNet: Simple policy network for DBlock models (legacy)
- FlexiblePolicyNet: Advanced policy network with MMW support and state normalization
- BlockValueNet: Value function network for Bellman residual methods

MMW JME '21 Support
===================

This module provides complete support for reproducing MMW JME '21 models:

1. **Dual Output Networks**: Automatic sigmoid + exponential transformations for
   consumption shares ζ ∈ [0,1] and Lagrange multipliers h > 0

2. **State Normalization**: MMW-style scaling to [-1,1] range for neural network inputs

3. **Configurable Architecture**: Exact MMW specifications (32×32×32, ReLU activation)

4. **Constraint Integration**: Built-in Fischer-Burmeister penalty support

Example Usage
=============

Basic Policy Network:
```python
from skagent.ann import FlexiblePolicyNet

# Create MMW-compatible policy network
policy_net = FlexiblePolicyNet(
    block=economic_model,
    width=32,  # MMW specification
    n_layers=3,  # MMW specification
    activation="relu",  # MMW specification
    transform=["sigmoid", "exp"],  # MMW dual outputs (ζ, h)
    # DBlock information set determines inputs
)
```

Value Function Network:
```python
from skagent.ann import BlockValueNet

# Create value function network for Bellman methods
value_net = BlockValueNet(
    block=economic_model, state_variables=["assets", "income"], width=32
)
```

References
==========

Maliar, L., Maliar, S., & Winant, P. (2021). Deep learning for solving dynamic
economic models. Journal of Monetary Economics, 122, 76-101.
"""

from skagent.grid import Grid
import typing  # at the top of the file ensure typing imported
from typing import Optional, Iterator, Callable, Union

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# input_tensor = input_tensor.to(device)

##########
# Constructing Nets


class Net(torch.nn.Module):
    """
    A simple feedforward neural network.

    Parameters
    ----------
    n_inputs : int
        Number of input features
    n_outputs : int
        Number of output features
    width : int, optional
        Width of hidden layers. Default is 32.
    """

    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(self, n_inputs, n_outputs, width):
        super(Net, self).__init__()

        self.hidden1 = torch.nn.Linear(n_inputs, width)
        self.hidden2 = torch.nn.Linear(width, width)
        self.output = torch.nn.Linear(width, n_outputs)

        self.to(device)

    def forward(self, x):
        # using the swish
        x = torch.nn.functional.silu(self.hidden1(x))
        x = torch.nn.functional.silu(self.hidden2(x))
        x = self.output(x)
        return x


class BlockPolicyNet(Net):
    """
    A neural network that implements a policy function for a given block.

    This is the original simple implementation for backward compatibility.
    For advanced features, use FlexiblePolicyNet instead.

    Parameters
    ----------
    block : DBlock
        The economic model block
    width : int, optional
        Width of hidden layers. Default is 32.
    """

    def __init__(self, block, width=32):
        self.block = block

        # pseudo -- assume only one for now
        control = self.block.dynamics[self.block.get_controls()[0]]

        super().__init__(len(control.iset), 1, width)

    def decision_function(self, states_t, shocks_t, parameters):
        """
        A decision function, from states, shocks, and parameters,
        to control variable values.

        Parameters
        ----------
        states_t: dict
            symbols : values

        shocks_t: dict
            symbols: values

        parameters : dict
            symbols : values

        Returns
        -------
        decisions - dict
            symbols : values
        """
        if parameters is None:
            parameters = {}
        vals = parameters | states_t | shocks_t

        # hacky -- should be moved into transition method as other option
        # very brittle, because it can interfere with constraints
        drs = {csym: lambda: 1 for csym in self.block.get_controls()}

        post = self.block.transition(vals, drs)

        # assuming only on control for now
        csym = self.block.get_controls()[0]
        control = self.block.dynamics[csym]

        # the inputs to the network are the information set of the control variable
        # The use of torch.stack and .T here are wild guesses, probably doesn't generalize

        iset_vals = [post[isym].flatten() for isym in control.iset]

        input_tensor = torch.stack(iset_vals).T

        input_tensor = input_tensor.to(device)
        output = self(input_tensor)  # application of network

        # again, assuming only one for now...
        # decisions = dict(zip([csym], output))
        # ... when using multiple csyms, note the orientation of the output tensor
        decisions = {csym: output.flatten()}
        return decisions

    def get_decision_function(self):
        def df(states_t, shocks_t, parameters):
            return self.decision_function(states_t, shocks_t, parameters)

        return df


class BlockValueNet(Net):
    """
    Neural network for value function approximation in the MMW JME '21 framework.

    This class implements a feedforward neural network V(s; θ) that approximates
    the value function for dynamic economic models. It is specifically designed
    for use with the Bellman residual loss function approach, where both policy
    and value functions are jointly trained.

    Network Architecture:
    - Input: State variables s_t (as specified in state_variables list)
    - Hidden layers: 2 layers with SiLU (Swish) activation
    - Output: Single scalar value V(s_t)

    Mathematical Role:
    In the Bellman equation V(s_t) = u(s_t, c_t) + β * E[V(s_{t+1})], this
    network provides the V(·) approximation used for both current and
    continuation value computations.

    Attributes
    ----------
    block : DBlock
        The economic model block (stored for reference)
    state_variables : list of str
        Names of state variables that define the input space
    """

    def __init__(self, block, state_variables, width=32):
        """
        Initialize the value function neural network.

        Parameters
        -----------
        block : DBlock
            The economic model block containing the problem specification.
            Used primarily for reference and potential future extensions.
        state_variables : list of str
            List of state variable names that define the value function's domain.
            These variables must be present in training grids and determine
            the network's input dimension.
            Example: ['assets', 'permanent_income'] for consumption problem
        width : int, optional
            Width of hidden layers in the neural network. Default is 32.
            Larger values increase approximation capacity but training cost.
        """
        self.block = block
        self.state_variables = state_variables

        # Value function takes state variables as input, outputs single value
        super().__init__(len(state_variables), 1, width)

    def value_function(self, states_t, parameters=None):
        """
        Compute value function V(s) for given states.

        Parameters
        -----------
        states_t : dict
            Dictionary mapping state variable names to tensor values
        parameters : dict, optional
            Model parameters (currently unused but kept for consistency)

        Returns
        --------
        torch.Tensor
            Value function values V(s)
        """
        if parameters is None:
            parameters = {}

        # Extract state values in correct order
        state_vals = [states_t[var].flatten() for var in self.state_variables]

        input_tensor = torch.stack(state_vals).T
        input_tensor = input_tensor.to(device)

        values = self(input_tensor)  # V(s)
        return values.flatten()

    def get_value_function(self):
        def vf(states_t, parameters=None):
            return self.value_function(states_t, parameters)

        return vf


################
# Model bindings


###########
# Training Nets


# General loss function that operates on tensor and averages over samples
def aggregate_net_loss(inputs: Grid, df, loss_function):
    """
    Compute a loss function over a tensor of inputs, given a decision function df.
    Return the mean.
    """
    # we include the network as a potential input to the loss function
    losses = loss_function(df, inputs)
    if hasattr(losses, "to"):  # slow, clumsy
        losses = losses.to(device)
    return losses.mean()


# Fix train_block_policy_nn typing
from collections.abc import Iterator as _Iterator

def train_block_policy_nn(
    block_policy_nn,
    inputs: Union[Grid, _Iterator[Grid]],
    loss_function: Callable[[Callable, Grid], torch.Tensor],
    *,
    epochs: int = 50,
    batch_size: Optional[int] = None,
    lr: float = 1e-3,
    lr_decay: Optional[float] = None,
):
    """Train a policy network with a given residual loss.

    Parameters
    ----------
    block_policy_nn : FlexiblePolicyNet | BlockPolicyNet
        Neural-network policy.
    inputs : Grid | Iterator[Grid]
        Either a fixed ``Grid`` (deterministic training) **or** a generator /
        iterator that yields fresh ``Grid`` batches (stochastic training).
    loss_function : callable
        Callable with signature ``loss_fn(decision_function, grid)`` returning a
        scalar tensor.
    epochs : int, default 50
        Number of training epochs.
    batch_size : int | None, optional
        If *inputs* is a ``Grid`` **and** *batch_size* is given, a random
        subset of that size is drawn each epoch (mini-batch training).  Ignored
        when *inputs* is a generator.
    lr : float, default 1e-3
        Adam learning-rate.
    lr_decay : float | None, optional
        If supplied, an exponential decay factor applied at the end of every
        epoch via ``torch.optim.lr_scheduler.ExponentialLR``.
    """

    # Optimiser and optional scheduler -------------------------------------------------
    optimiser = torch.optim.Adam(block_policy_nn.parameters(), lr=lr)
    scheduler = (
        torch.optim.lr_scheduler.ExponentialLR(optimiser, lr_decay)
        if lr_decay is not None
        else None
    )

    block_policy_nn._training_losses: list[float] = []  # type: ignore[attr-defined]

    # Helper ---------------------------------------------------------------------------
    def _sample_mini(grid: Grid, n: Optional[int]) -> Grid:
        if n is None or n >= len(grid.values):
            return grid
        idx = torch.randint(0, len(grid.values), (n,))
        return Grid(grid.labels, grid.values[idx])

    # -----------------------------------------------------------------------------
    for epoch in range(epochs):
        optimiser.zero_grad()
        if isinstance(inputs, Grid):
            batch = _sample_mini(inputs, batch_size)
        else:
            batch = next(inputs)
        loss = loss_function(block_policy_nn.get_decision_function(), batch)
        loss.backward()
        optimiser.step()
        if scheduler is not None:
            scheduler.step()
        block_policy_nn._training_losses.append(float(loss.detach()))
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    return block_policy_nn


def train_bellman_nets(policy_net, value_net, inputs, loss_function, epochs=50):
    """
    Joint training of policy and value networks for MMW JME '21 Bellman approach.

    This function implements the joint optimization procedure for the Bellman
    residual loss approach described in MMW JME '21 Section 2.4. Unlike the
    primary EDLR method which only trains a policy network, this approach
    simultaneously optimizes both policy φ(s,ε; θ_π) and value V(s; θ_v) networks
    to minimize Bellman equation violations.

    Training Procedure:
    1. Forward pass: Compute Bellman residuals for each grid point
    2. Backward pass: Gradients flow through both networks simultaneously
    3. Parameter update: Joint Adam optimization of θ_π and θ_v

    Mathematical Objective:
    min_{θ_π, θ_v} E[|V(s_t; θ_v) - [u(s_t, φ(s_t,ε_t; θ_π)) + β*E[V(s_{t+1}; θ_v)]]|²]

    Convergence Properties:
    - Joint optimization can be more stable than alternating methods
    - Bellman residuals provide direct measure of solution quality
    - Training typically converges faster than EDLR for well-posed problems

    Parameters
    -----------
    policy_net : BlockPolicyNet
        Policy function approximation network φ(s,ε; θ_π).
        Network parameters θ_π will be optimized during training.
    value_net : BlockValueNet
        Value function approximation network V(s; θ_v).
        Network parameters θ_v will be optimized during training.
    inputs : Grid
        Training data grid containing state variables and shock realizations.
        Must have the 2-shock structure from generate_bellman_training_grid().
    loss_function : callable
        Bellman residual loss function from get_bellman_residual_loss().
        Should have signature: (policy_function, value_function, grid) -> tensor
    epochs : int, optional
        Number of training epochs. Default is 50.
        More epochs may be needed for complex problems or large networks.

    Returns
    --------
    tuple of (BlockPolicyNet, BlockValueNet)
        The trained policy and value networks with optimized parameters.
        Networks are modified in-place, but also returned for convenience.

    Notes
    -----
    - Uses Adam optimizer with learning rate 0.01
    - Prints training loss every 100 epochs for monitoring
    - Both networks are modified in-place during training
    - Requires CUDA-compatible loss function for GPU training
    """
    # Optimize both networks jointly
    optimizer = torch.optim.Adam(
        list(policy_net.parameters()) + list(value_net.parameters()), lr=0.01
    )

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Loss function expects (policy_function, value_function, input_grid)
        loss = loss_function(
            policy_net.get_decision_function(), value_net.get_value_function(), inputs
        )

        if hasattr(loss, "to"):
            loss = loss.to(device)
        loss = loss.mean()

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(
                "Epoch {}: Bellman Loss = {}".format(epoch, loss.cpu().detach().numpy())
            )

    return policy_net, value_net


class FlexibleNet(torch.nn.Module):
    """
    A flexible feedforward neural network with configurable architecture.

    Parameters
    ----------
    n_inputs : int
        Number of input features
    n_outputs : int
        Number of output features
    width : int, optional
        Width of hidden layers. Default is 32.
    n_layers : int, optional
        Number of hidden layers (1-10). Default is 2.
    activation : str, list, callable, or None, optional
        Activation function(s) to use. Options:
        - str: Apply same activation to all layers ('silu', 'relu', 'tanh', 'sigmoid')
        - list: Apply different activations to each layer, e.g., ['relu', 'tanh', 'silu']
        - callable: Custom activation function
        - None: No activation (identity function)

        Available activations: 'silu', 'relu', 'tanh', 'sigmoid', 'identity'
        Default is 'silu'.
    transform : str, list, callable, or None, optional
        Transformation to apply to outputs. Options:
        - str: Apply same transform to all outputs ('sigmoid', 'exp', 'tanh', etc.)
        - list: Apply different transforms to each output, e.g., ['sigmoid', 'exp']
        - callable: Custom transformation function
        - None: No transformation

        Available transforms: 'sigmoid', 'exp', 'tanh', 'relu', 'softplus', 'softmax', 'abs', 'square', 'identity'
        Default is None.
    """

    def __init__(
        self,
        n_inputs,
        n_outputs,
        width=32,
        n_layers=2,
        activation="silu",
        transform=None,
        init_seed=None,
        copy_weights_from=None,
    ):
        super().__init__()

        # Validate n_layers
        if not (1 <= n_layers <= 10):
            raise ValueError(f"n_layers must be between 1 and 10, got {n_layers}")

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.width = width
        self.n_layers = n_layers
        self.transform = transform
        self.init_seed = init_seed
        self.copy_weights_from = copy_weights_from

        # Set activation function(s) and track which are identity for performance
        if isinstance(activation, list):
            if len(activation) != n_layers:
                raise ValueError(
                    f"Number of activations ({len(activation)}) must match "
                    f"number of layers ({n_layers})"
                )
            self.activations = [self._get_activation_fn(act) for act in activation]
            self.activation_is_identity = [self._is_identity(act) for act in activation]
        else:
            # Single activation applied to all layers
            self.activations = [self._get_activation_fn(activation)] * n_layers
            self.activation_is_identity = [self._is_identity(activation)] * n_layers

        # Build network layers
        self.layers = torch.nn.ModuleList()

        # First hidden layer
        self.layers.append(torch.nn.Linear(n_inputs, width))

        # Additional hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(torch.nn.Linear(width, width))

        # Output layer
        self.output = torch.nn.Linear(width, n_outputs)

        # Initialize weights first (before device placement)
        if init_seed is not None:
            # Save current random state
            current_state = torch.get_rng_state()
            # Set seed for initialization
            torch.manual_seed(init_seed)
        
        # Custom weight initialisation to match MMW notebook (normal std=0.05)
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.05)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        if init_seed is not None:
            # Restore previous random state
            torch.set_rng_state(current_state)

        # Copy weights AFTER initialization if requested
        if copy_weights_from is not None:
            self._copy_weights_from_network(copy_weights_from)

        # Move to device for backward compatibility (after all initialization)
        self.to(device)

    def _copy_weights_from_network(self, source_network):
        """Copy weights from another network with compatible architecture."""
        source_params = list(source_network.parameters())
        target_params = list(self.parameters())
        
        if len(source_params) != len(target_params):
            raise ValueError(f"Network architectures incompatible: {len(source_params)} vs {len(target_params)} parameters")
        
        with torch.no_grad():
            for target_param, source_param in zip(target_params, source_params):
                if target_param.shape != source_param.shape:
                    raise ValueError(f"Parameter shape mismatch: {target_param.shape} vs {source_param.shape}")
                target_param.copy_(source_param)

    def _get_activation_fn(self, activation):
        """Get activation function from string name, callable, or None."""
        if activation == "silu":
            return torch.nn.functional.silu
        elif activation == "relu":
            return torch.nn.functional.relu
        elif activation == "tanh":
            return torch.nn.functional.tanh
        elif activation == "sigmoid":
            return torch.nn.functional.sigmoid
        elif activation == "identity" or activation is None:
            return None  # Will be skipped in forward pass for performance
        elif callable(activation):
            return activation
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def _is_identity(self, activation):
        """Check if activation is identity/None for performance optimization."""
        return activation == "identity" or activation is None

    @property
    def device(self):
        """Device property for backward compatibility."""
        return next(self.parameters()).device

    def forward(self, x):
        # Forward through hidden layers with layer-specific activations
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Skip identity activations for performance
            if not self.activation_is_identity[i]:
                x = self.activations[i](x)

        # Output layer
        x = self.output(x)

        # Apply output transformation if specified
        if self.transform is not None:
            x = self._apply_transform(x)

        return x

    def _apply_transform(self, x):
        """Apply output transformation based on configuration."""
        if isinstance(self.transform, list):
            # List of transforms: apply each transform to corresponding output
            if len(self.transform) != x.shape[-1]:
                raise ValueError(
                    f"Number of transforms ({len(self.transform)}) must match "
                    f"number of outputs ({x.shape[-1]})"
                )

            transformed_outputs = []
            for i, transform in enumerate(self.transform):
                output_i = x[..., i]
                transformed_outputs.append(
                    self._apply_single_transform(output_i, transform)
                )

            return torch.stack(transformed_outputs, dim=-1)

        elif self.transform is None:
            # No transformation
            return x

        else:
            # Single transform applied to all outputs (string or callable)
            return self._apply_single_transform(x, self.transform)

    def _apply_single_transform(self, x, transform):
        """Apply a single transformation to a tensor."""
        if transform == "sigmoid":
            return torch.sigmoid(x)
        elif transform == "exp":
            return torch.exp(x)
        elif transform == "tanh":
            return torch.tanh(x)
        elif transform == "relu":
            return torch.nn.functional.relu(x)
        elif transform == "softplus":
            return torch.nn.functional.softplus(x)
        elif transform == "softmax":
            return torch.nn.functional.softmax(x, dim=-1)
        elif transform == "abs":
            return torch.abs(x)
        elif transform == "square":
            return x**2
        elif transform == "identity" or transform is None:
            return x
        elif callable(transform):
            return transform(x)
        else:
            raise ValueError(f"Unknown single transform: {transform}")


class FlexiblePolicyNet(FlexibleNet):
    """
    A flexible neural network that implements a policy function for a given block.

    This network automatically handles input/output dimensions based on the block
    specification. Inherits from FlexibleNet to provide configurable architecture
    with economic model integration.

    The network relies solely on the DBlock specification to determine its inputs.
    For example, if the control variable's information set is ['r','delta','p','q','w'],
    those symbols will be stacked (after an inexpensive transition call) and fed to the
    neural network. No additional flags are necessary.

    Parameters
    ----------
    block : DBlock
        The economic model block
    width : int, optional
        Width of hidden layers. Default is 32.
    n_layers : int, optional
        Number of hidden layers (1-10). Default is 2.
    activation : str, list, callable, or None, optional
        Activation function(s). Default is 'silu'.
    transform : str, list, callable, or None, optional
        Output transformation to apply. Options:
        - str: Apply same transform to all outputs ('sigmoid', 'exp', 'tanh', etc.)
        - list: Apply different transforms to each output, e.g., ['sigmoid', 'exp'] for MMW-style outputs
        - callable: Custom transformation function
        - None: No transformation

        Available transforms: 'sigmoid', 'exp', 'tanh', 'relu', 'softplus', 'softmax', 'abs', 'square', 'identity'
        Default is None.
    """

    def __init__(
        self,
        block,
        width=32,
        n_layers=2,
        activation="silu",
        transform=None,
        init_seed=None,
        copy_weights_from=None,
    ):
        self.block = block

        # Get control variables
        controls = block.get_controls()
        self.control_names = controls

        # Group controls by their information set for efficient computation
        self.control_groups = self._group_controls_by_iset()

        # Use first control's information set to determine input dimension
        control = self.block.dynamics[self.block.get_controls()[0]]
        n_inputs = len(control.iset)

        # Determine number of outputs based on controls and transforms
        if isinstance(transform, list):
            n_outputs = len(transform)
        else:
            n_outputs = len(controls)

        # Initialize parent FlexibleNet with computed parameters
        super().__init__(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            width=width,
            n_layers=n_layers,
            activation=activation,
            transform=transform,
            init_seed=init_seed,
            copy_weights_from=copy_weights_from,
        )

        # Store for constraint handling and auxiliary output access
        self.last_outputs = None
        self.last_raw_outputs = None
        self.last_states = None

    def _group_controls_by_iset(self):
        """
        Group controls by their information set for efficient computation.

        Returns
        --------
        dict
            Dictionary mapping frozenset(iset) -> list of control names
        """
        groups = {}
        for control_name in self.control_names:
            control = self.block.dynamics[control_name]
            iset_key = frozenset(control.iset)
            if iset_key not in groups:
                groups[iset_key] = []
            groups[iset_key].append(control_name)
        return groups

    def get_auxiliary_outputs(self):
        """
        Get auxiliary outputs for constraint functions.

        Returns
        --------
        dict
            Dictionary with 'raw', 'transformed', 'states' keys
        """
        return {
            "raw": self.last_raw_outputs,
            "transformed": self.last_outputs,
            "states": self.last_states,
            "controls": dict(
                zip(
                    self.control_names,
                    [
                        self.last_outputs[..., i]
                        if self.last_outputs is not None
                        else None
                        for i in range(len(self.control_names))
                    ],
                )
            ),
        }

    def forward(self, states, shocks, parameters):
        """
        Forward pass through the policy network.

        Parameters
        -----------
        states : dict
            Dictionary mapping state variable names to tensor values
        shocks : dict
            Dictionary mapping shock variable names to tensor values
        parameters : dict
            Model calibration parameters

        Returns
        --------
        dict
            Dictionary mapping control variable names to policy outputs
        """
        if parameters is None:
            parameters = {}

        # Fast-path: if no shocks and all inputs already in `states`, skip expensive transition
        control_name = self.control_names[0]
        control = self.block.dynamics[control_name]

        if not shocks and all(var in states for var in control.iset):
            post = states  # already have required inputs
        else:
            vals = parameters | states | shocks
            drs = {csym: lambda: 1 for csym in self.block.get_controls()}
            post = self.block.transition(vals, drs)

        # Extract input values from control's information set
        input_vals = []
        for var in control.iset:
            if var in post:
                val = post[var]
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val, dtype=torch.float32)
                input_vals.append(val.flatten())
            else:
                raise ValueError(f"Variable {var} not found in post-transition values")

        if not input_vals:
            raise ValueError("No input variables found from control information set")

        input_tensor = torch.stack(input_vals).T.to(self.device)

        # ------- Forward pass WITHOUT transforms first -------
        saved_transform = self.transform
        self.transform = None
        raw_outputs = super().forward(input_tensor)
        self.transform = saved_transform

        # Apply per-output transform AFTER slicing for list case
        if isinstance(self.transform, list):
            if len(self.transform) != raw_outputs.shape[-1]:
                raise ValueError("Transform list length mismatch with outputs")
            transformed = []
            for i, tr in enumerate(self.transform):
                transformed.append(self._apply_single_transform(raw_outputs[..., i], tr))
            outputs = torch.stack(transformed, dim=-1)
        else:
            # single or None
            outputs = self._apply_single_transform(raw_outputs, self.transform)

        self.last_raw_outputs = raw_outputs
        self.last_outputs = outputs
        self.last_states = {**states, **shocks}

        # Return control outputs following skagent pattern
        if len(self.control_names) == 1:
            return {self.control_names[0]: outputs.flatten()}
        else:
            # Multiple controls - map outputs to control names
            result = {}
            for i, control_name in enumerate(self.control_names):
                if i < outputs.shape[-1]:
                    result[control_name] = outputs[..., i].flatten()
                else:
                    result[control_name] = torch.zeros_like(outputs[..., 0].flatten())
            return result

    def get_decision_function(self):
        """Return a decision function compatible with skagent interfaces."""

        def decision_function(states_t, shocks_t, parameters):
            return self.decision_function(states_t, shocks_t, parameters)

        return decision_function

    def decision_function(self, states_t, shocks_t, parameters):
        """
        Decision function interface for compatibility with skagent.

        Parameters
        -----------
        states_t : dict
            State variables at time t
        shocks_t : dict
            Shock realizations at time t
        parameters : dict
            Model parameters

        Returns
        --------
        dict
            Control variable decisions
        """
        return self.forward(states_t, shocks_t, parameters)
