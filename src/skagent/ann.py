from skagent.grid import Grid
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# input_tensor = input_tensor.to(device)

##########
# Constructing Nets


class Net(torch.nn.Module):
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
    def __init__(self, block, width=32):
        self.block = block

        ## pseudo -- assume only one for now
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
        decisions = {csym: output.flatten()}
        return decisions

    def get_decision_function(self):
        def df(states_t, shocks_t, parameters):
            return self.decision_function(states_t, shocks_t, parameters)

        return df


class BlockValueNet(Net):
    """
    A neural network for approximating value functions in dynamic programming problems.

    This network takes state variables as input and outputs value estimates.
    It's designed to work with the Bellman equation loss functions in the Maliar method.
    """

    def __init__(self, block, width: int = 32):
        """
        Initialize the BlockValueNet.

        Parameters
        ----------
        block : model.DBlock
            The model block containing state variables and dynamics
        width : int, optional
            Width of hidden layers, by default 32
        """
        self.block = block

        # Value function should use the same information set as the policy function
        # Both V(s) and π(s) take the same state information as input
        ## pseudo -- assume only one control for now (same as BlockPolicyNet)
        control = self.block.dynamics[self.block.get_controls()[0]]

        # Use the same information set as the policy network
        self.state_variables = sorted(list(control.iset))

        # Value function takes state variables as input and outputs a scalar value
        super().__init__(len(self.state_variables), 1, width)

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
    block_policy_nn, inputs: Grid, loss_function: callable, epochs=50
):
    ## to change
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
    ## to change
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
    ## to change
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
                transformed.append(
                    self._apply_single_transform(raw_outputs[..., i], tr)
                )
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
