import inspect
from skagent.grid import Grid
import torch
from skagent.utils import create_vectorized_function_wrapper_with_mapping
from typing import Callable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# input_tensor = input_tensor.to(device)

##########
# Constructing Nets


class Net(torch.nn.Module):
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
            raise ValueError(
                f"Network architectures incompatible: {len(source_params)} vs {len(target_params)} parameters"
            )

        with torch.no_grad():
            for target_param, source_param in zip(target_params, source_params):
                if target_param.shape != source_param.shape:
                    raise ValueError(
                        f"Parameter shape mismatch: {target_param.shape} vs {source_param.shape}"
                    )
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


class BlockPolicyNet(Net):
    """
    A neural network for policy functions in dynamic programming problems.

    This network inherits from Net and provides economic model integration.
    It automatically determines input/output dimensions from the model block specification
    and handles control variable bounds.

    Parameters
    -----------
    block : model.DBlock
        The model block containing control variables and dynamics
    apply_open_bounds : bool, optional
        If True, then the network forward output is normalized by the upper and/or lower bounds,
        computed as a function of the input tensor. These bounds are "open" because output
        can be arbitrarily close to, but not equal to, the bounds. Default is True.
    width : int, optional
        Width of hidden layers. Default is 32.
    n_layers : int, optional
        Number of hidden layers (1-10). Default is 2.
    activation : str, list, callable, or None, optional
        Activation function(s). See Net documentation for details. Default is 'silu'.
    transform : str, list, callable, or None, optional
        Output transformation. See Net documentation for details. Default is None.
    **kwargs
        Additional keyword arguments passed to Net. See Net class
        documentation for all available options including init_seed, copy_weights_from, etc.
    """

    def __init__(self, block, apply_open_bounds=True, width=32, **kwargs):
        self.block = block
        self.apply_open_bounds = apply_open_bounds

        # pseudo -- assume only one for now
        # assuming only on control for now
        self.csym = self.block.get_controls()[0]
        self.control = self.block.dynamics[self.csym]
        self.iset = self.control.iset

        # assess whether/how the control is bounded
        # this will be more challenging with multiple controls.
        # If not None, these will be _functions_.
        # If it is bounded, set up the vectorized version of the bound
        # This will be used directly in the forward pass of the network.
        self.upper_bound = self.control.upper_bound
        self.upper_bound_vec_func, self.upper_bound_param_to_column = self._setup_bound(
            self.upper_bound, "Upper bound"
        )
        self.lower_bound = self.control.lower_bound
        self.lower_bound_vec_func, self.lower_bound_param_to_column = self._setup_bound(
            self.lower_bound, "Lower bound"
        )

        super().__init__(n_inputs=len(self.iset), n_outputs=1, width=width, **kwargs)

    def _setup_bound(self, bound_func, bound_name):
        if bound_func:
            sig = inspect.signature(bound_func)
            param_names = list(sig.parameters.keys())
            param_to_column = {}
            for param_name in param_names:
                if param_name in self.iset:
                    param_to_column[param_name] = self.iset.index(param_name)
                else:
                    raise ValueError(
                        f"{bound_name} parameter '{param_name}' not found in control.iset: {self.iset}"
                    )
            vec_func = create_vectorized_function_wrapper_with_mapping(
                bound_func, param_to_column
            )
            return vec_func, param_to_column
        return None, None

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

        post = self.block.transition(vals, drs, until=self.csym)

        # the inputs to the network are the information set of the control variable
        # The use of torch.stack and .T here are wild guesses, probably doesn't generalize
        iset_vals = [post[isym].flatten() for isym in self.iset]
        if len(iset_vals) > 0:
            input_tensor = torch.stack(iset_vals).T
            input_tensor = input_tensor.to(device)
        else:
            batch_size = len(next(iter(states_t.values())))
            input_tensor = torch.empty(batch_size, 0, device=device)

        output = self(input_tensor)  # application of network

        # again, assuming only one for now...
        # decisions = dict(zip([csym], output))
        # ... when using multiple csyms, note the orientation of the output tensor
        decisions = {self.csym: output.flatten()}
        return decisions

    def forward(self, x):
        """
        Note that this uses the same architecture of the superclass
        but adds on a normalization layer appropriate to the
        bounds of the decision rule.
        """

        # using the swish
        x1 = super().forward(x)

        if self.apply_open_bounds:
            if not self.upper_bound and not self.lower_bound:
                x2 = x1
            elif self.upper_bound and self.lower_bound:
                # Compute bounds from input using wrapped functions
                upper_bound = self.upper_bound_vec_func(x)
                lower_bound = self.lower_bound_vec_func(x)

                # Scale to bounds
                x2 = lower_bound + torch.nn.functional.sigmoid(x1) * (
                    upper_bound - lower_bound
                )

            elif self.lower_bound and not self.upper_bound:
                lower_bound = self.lower_bound_vec_func(x)
                x2 = lower_bound + torch.nn.functional.softplus(x1)

            elif not self.lower_bound and self.upper_bound:
                upper_bound = self.upper_bound_vec_func(x)
                x2 = upper_bound - torch.nn.functional.softplus(x1)
        else:
            # return un-normalized reals.
            x2 = x1

        return x2

    def get_decision_function(self):
        def df(states_t, shocks_t, parameters):
            return self.decision_function(states_t, shocks_t, parameters)

        return df


class BlockValueNet(Net):
    """
    A neural network for approximating value functions in dynamic programming problems.

    This network takes state variables as input and outputs value estimates.
    It's designed to work with the Bellman equation loss functions in the Maliar method.
    Inherits from Net to provide configurable architecture.

    Parameters
    ----------
    block : model.DBlock
        The model block containing state variables and dynamics
    width : int, optional
        Width of hidden layers. Default is 32.
    n_layers : int, optional
        Number of hidden layers (1-10). Default is 2.
    activation : str, list, callable, or None, optional
        Activation function(s). See Net documentation for details. Default is 'silu'.
    transform : str, list, callable, or None, optional
        Output transformation. See Net documentation for details. Default is None.
    **kwargs
        Additional keyword arguments passed to Net. See Net class
        documentation for all available options including init_seed, copy_weights_from, etc.
    """

    def __init__(self, block, width: int = 32, **kwargs):
        """
        Initialize the BlockValueNet.
        """
        self.block = block

        # Value function should use the same information set as the policy function
        # Both V(s) and π(s) take the same state information as input
        # pseudo -- assume only one control for now (same as BlockPolicyNet)
        control = self.block.dynamics[self.block.get_controls()[0]]

        # Use the same information set as the policy network
        self.state_variables = sorted(list(control.iset))

        # Value function takes state variables as input and outputs a scalar value
        super().__init__(
            n_inputs=len(self.state_variables), n_outputs=1, width=width, **kwargs
        )

    def value_function(self, states_t, shocks_t={}, parameters={}):
        """
        Compute value function estimates for given state variables.

        The value function takes the same information as the policy function
        (the control's information set) but doesn't need to compute transitions.

        Parameters
        ----------
        states_t : dict
            State variables as dict (e.g., {"wealth": tensor(...)})
        shocks_t : dict, optional
            Shock variables as dict (not used but kept for interface consistency)
        parameters : dict, optional
            Model parameters (not used but kept for interface consistency)

        Returns
        -------
        torch.Tensor
            Value function estimates
        """
        # Get the control's information set (same as policy network)
        csym = self.block.get_controls()[0]
        control = self.block.dynamics[csym]

        # The inputs to the network are the information set variables
        # Combine states_t and shocks_t to get all available variables
        all_vars = states_t | shocks_t
        iset_vals = [all_vars[isym].flatten() for isym in control.iset]

        input_tensor = torch.stack(iset_vals).T

        # Keep tensor on same device as input (don't force to CUDA device)
        if hasattr(iset_vals[0], "device"):
            input_tensor = input_tensor.to(iset_vals[0].device)
            # Also move network to same device
            self.to(iset_vals[0].device)
        else:
            input_tensor = input_tensor.to(device)

        # Forward pass through network
        output = self(input_tensor)

        return output.flatten()

    def get_value_function(self):
        """
        Get a callable value function for use with loss functions.

        This follows the same pattern as BlockPolicyNet.get_decision_function()

        Returns
        -------
        callable
            A function that takes states, shocks, and parameters and returns value estimates
        """

        def vf(states_t, shocks_t={}, parameters={}):
            return self.value_function(states_t, shocks_t, parameters)

        return vf


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


def train_block_policy_nn(
    block_policy_nn, inputs: Grid, loss_function: Callable, epochs=50
):
    # to change
    # criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(block_policy_nn.parameters(), lr=0.01)  # Using Adam

    for epoch in range(epochs):
        running_loss = 0.0
        optimizer.zero_grad()
        loss = aggregate_net_loss(
            inputs, block_policy_nn.get_decision_function(), loss_function
        )
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if epoch % 100 == 0:
            print("Epoch {}: Loss = {}".format(epoch, loss.cpu().detach().numpy()))

    return block_policy_nn


def train_block_value_nn(block_value_nn, inputs: Grid, loss_function, epochs=50):
    """
    Train a BlockValueNet using a value function loss.

    Parameters
    ----------
    block_value_nn : BlockValueNet
        The value network to train
    inputs : Grid
        Input grid containing state variables
    loss_function : callable
        Loss function that takes (value_function, input_grid) and returns loss
    epochs : int, optional
        Number of training epochs, by default 50

    Returns
    -------
    BlockValueNet
        The trained value network
    """
    # to change
    # criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(block_value_nn.parameters(), lr=0.01)  # Using Adam

    for epoch in range(epochs):
        running_loss = 0.0
        optimizer.zero_grad()

        # Use aggregate_net_loss for consistency with policy training
        # For value networks, we pass the value function instead of decision function
        loss = aggregate_net_loss(
            inputs, block_value_nn.get_value_function(), loss_function
        )
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if epoch % 100 == 0:
            print("Epoch {}: Loss = {}".format(epoch, loss.cpu().detach().numpy()))

    return block_value_nn


def train_block_value_and_policy_nn(
    block_policy_nn,
    block_value_nn,
    inputs: Grid,
    policy_loss_function,
    value_loss_function,
    epochs=50,
):
    """
    Train both BlockPolicyNet and BlockValueNet jointly for value function iteration.

    This follows the same pattern as train_block_policy_nn and train_block_value_nn:
    takes existing networks and loss functions, trains them, returns trained networks.

    Parameters
    ----------
    block_policy_nn : BlockPolicyNet
        The policy network to train
    block_value_nn : BlockValueNet
        The value network to train
    inputs : Grid
        Input grid containing states and shocks
    policy_loss_function : callable
        Loss function for policy training (takes decision_function, input_grid)
    value_loss_function : callable
        Loss function for value training (takes value_function, input_grid)
    epochs : int, optional
        Number of training epochs, by default 50

    Returns
    -------
    tuple
        (trained_policy_nn, trained_value_nn)
    """
    # to change
    # criterion = torch.nn.MSELoss()
    policy_optimizer = torch.optim.Adam(
        block_policy_nn.parameters(), lr=0.01
    )  # Using Adam
    value_optimizer = torch.optim.Adam(
        block_value_nn.parameters(), lr=0.01
    )  # Using Adam

    for epoch in range(epochs):
        # Train policy network
        policy_optimizer.zero_grad()
        policy_loss = aggregate_net_loss(
            inputs, block_policy_nn.get_decision_function(), policy_loss_function
        )
        policy_loss.backward()
        policy_optimizer.step()

        # Train value network
        value_optimizer.zero_grad()
        value_loss = aggregate_net_loss(
            inputs, block_value_nn.get_value_function(), value_loss_function
        )
        value_loss.backward()
        value_optimizer.step()

        if epoch % 100 == 0:
            print(
                "Epoch {}: Policy Loss = {}, Value Loss = {}".format(
                    epoch,
                    policy_loss.cpu().detach().numpy(),
                    value_loss.cpu().detach().numpy(),
                )
            )

    return block_policy_nn, block_value_nn
