import inspect
import logging
from skagent.grid import Grid
import torch
from skagent.utils import create_vectorized_function_wrapper_with_mapping
from typing import Callable, Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BellmanPeriodMixin:
    """
    Mixin class providing common Bellman period initialization for Block*Net classes.

    This mixin extracts and stores the bellman period, control symbol, control object,
    and information set that are commonly needed across BlockPolicyNet and
    BlockPolicyValueNet.
    """

    # Contract: attributes/methods that concrete subclasses provide and that
    # the shared helpers below rely on. Declared (without assignment) so that
    # static analysis can see the mixin's dependencies.
    iset: list
    apply_open_bounds: bool
    upper_bound_vec_func: Optional[Callable]
    lower_bound_vec_func: Optional[Callable]

    def decision_function(self, states_t, shocks_t, parameters) -> dict:
        """Map states, shocks, and parameters to a controls dict.

        Implemented by each concrete network; declared here as the contract
        that :meth:`get_decision_function` closes over.
        """
        raise NotImplementedError

    def _init_bellman_period(self, bellman_period, control_sym=None):
        """
        Initialize bellman period related attributes.

        Parameters
        ----------
        bellman_period : BellmanPeriod
            The model Bellman Period
        control_sym : str, optional
            The symbol for the control variable. If None, uses the first control.
        """
        self.bellman_period = bellman_period

        # Get the control symbol (assume only one for now)
        if control_sym is None:
            control_sym = next(iter(self.bellman_period.get_controls()))

        self.control_sym = control_sym
        self.cobj = self.bellman_period.block.dynamics[control_sym]
        # Snapshot the information set so later mutation of cobj.iset cannot
        # silently desync the network's input dimension and bound mappings.
        self.iset = list(self.cobj.iset)

    def _setup_bound(self, bound_func, bound_name):
        """Set up a vectorized bound function from a callable or None.

        Numeric bounds (e.g. ``lower_bound=0.0``) must be supplied as
        zero-argument callables (e.g. ``lambda: 0.0``); a plain numeric value
        is rejected so it cannot silently disable the bound through a
        truthiness check.
        """
        if bound_func is None:
            return None, None
        if not callable(bound_func):
            raise TypeError(
                f"{bound_name} must be a callable or None; got "
                f"{type(bound_func).__name__}. Wrap numeric constants in a "
                f"zero-argument callable, e.g. `lambda: {bound_func}`."
            )
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

    def _apply_open_bounds(self, x1, x_input):
        """Scale raw network output ``x1`` into the control's open bounds.

        "Open" means the output can approach but never equal a bound. The
        branch is selected from which bound vec-funcs are present (probed
        via ``is None`` so a numeric bound such as ``0.0`` is not treated as
        absent):

        - no bounds: identity
        - both bounds: ``lower + sigmoid(x1) * (upper - lower)``
        - lower only: ``lower + softplus(x1)``
        - upper only: ``upper - softplus(x1)``

        The caller is responsible for supplying bound callables with
        ``lower < upper``; an inverted or degenerate pair yields a constant
        or out-of-range policy with no error raised here (a per-call check
        is omitted to keep the forward pass free of host-device syncs).
        """
        if not self.apply_open_bounds:
            return x1
        # Bind to locals so static narrowing of the `is not None` checks
        # carries through to the calls below.
        upper = self.upper_bound_vec_func
        lower = self.lower_bound_vec_func
        if upper is None and lower is None:
            return x1
        if upper is not None and lower is not None:
            ub = upper(x_input)
            lb = lower(x_input)
            return lb + torch.nn.functional.sigmoid(x1) * (ub - lb)
        if lower is not None:
            return lower(x_input) + torch.nn.functional.softplus(x1)
        # upper only
        ub = upper(x_input)
        return ub - torch.nn.functional.softplus(x1)

    def get_decision_function(self):
        """Return a closure ``(states, shocks, parameters) -> controls dict``."""

        def df(states_t, shocks_t, parameters):
            return self.decision_function(states_t, shocks_t, parameters)

        return df


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


class BlockPolicyNet(BellmanPeriodMixin, Net):
    """
    A neural network for policy functions in dynamic programming problems.

    This network wraps a :class:`Net` and integrates with the
    :class:`~skagent.bellman.BellmanPeriod` interface. It automatically
    determines input/output dimensions from the model block specification
    and enforces control variable bounds.

    Parameters
    -----------
    bellman_period : BellmanPeriod
        The model Bellman Period
    apply_open_bounds : bool, optional
        If True, then the network forward output is normalized by the upper and/or lower bounds,
        computed as a function of the input tensor. These bounds are "open" because output
        can be arbitrarily close to, but not equal to, the bounds. Default is True.
    control_sym : string, optional
        The symbol for the control variable.
    width : int, optional
        Width of hidden layers. Default is 32.
    **kwargs
        Additional keyword arguments passed to Net. See Net class
        documentation for all available options including activation, transform, n_layers, init_seed, copy_weights_from, etc.
    """

    def __init__(
        self,
        bellman_period,
        control_sym=None,
        apply_open_bounds=True,
        width=32,
        **kwargs,
    ):
        self._init_bellman_period(bellman_period, control_sym)
        self.apply_open_bounds = apply_open_bounds

        ## assess whether/how the control is bounded
        # this will be more challenging with multiple controls.
        # If not None, these will be _functions_.
        # If it is bounded, set up the vectorized version of the bound
        # This will be used directly in the forward pass of the network.
        self.upper_bound = self.cobj.upper_bound
        self.upper_bound_vec_func, self.upper_bound_param_to_column = self._setup_bound(
            self.upper_bound, "Upper bound"
        )
        self.lower_bound = self.cobj.lower_bound
        self.lower_bound_vec_func, self.lower_bound_param_to_column = self._setup_bound(
            self.lower_bound, "Lower bound"
        )

        super().__init__(n_inputs=len(self.iset), n_outputs=1, width=width, **kwargs)

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
        iset_dict = self.bellman_period.compute_pre_state(
            self.control_sym, states_t, shocks=shocks_t, parameters=parameters
        )
        # Stack iset values as rows, then transpose to shape (n_samples, n_iset)
        # for batch matrix operations.
        iset_vals = [iset_dict[isym].flatten() for isym in self.iset]

        def get_tensor_size(d):
            for value in d.values():
                if hasattr(value, "numel"):  # PyTorch tensor
                    return value.numel()
                elif hasattr(value, "size"):  # NumPy array or other array-like
                    return value.size
            return 1  # No tensors found

        output = self.get_decision_rule(length=get_tensor_size(iset_dict))[
            self.control_sym
        ](*iset_vals)

        decisions = {self.control_sym: output}
        return decisions

    def forward(self, x):
        """
        Note that this uses the same architecture of the superclass
        but adds on a normalization layer appropriate to the
        bounds of the decision rule.
        """
        x1 = super().forward(x)
        return self._apply_open_bounds(x1, x)

    def get_core_function(self, length=None):
        return self.get_decision_rule(length=length)

    def get_decision_rule(self, length=None):
        """
        Returns the decision rule corresponding to this neural network.
        """

        def decision_rule(*information):
            """
            A decision rule positional arguments (reflecting the information set)
            values to control values.

            Parameters
            ----------
            information: *args
                values arrays

            Returns
            -------

            decisions - array
            """
            if len(information) > 0:
                input_tensor = torch.stack(information).T
                input_tensor = input_tensor.to(device)
            else:
                batch_size = length

                if batch_size is None:
                    raise Exception(
                        "You must pass a tensor length when creating a decision rule"
                        " with an empty information set."
                    )
                input_tensor = torch.empty(batch_size, 0, device=device)

            return self(input_tensor).flatten()  # application of network

        return {self.control_sym: decision_rule}


class BlockPolicyValueNet(BellmanPeriodMixin, Net):
    """
    Single neural network with shared backbone for both policy and value.

    Architecture: shared hidden layers → two output heads:
    - **Policy head** — bounded output (sigmoid-scaled to satisfy constraints)
    - **Value head** — unconstrained scalar output

    Sharing the backbone means one optimizer updates all weights
    simultaneously, and the value head anchors the control *level* that
    first-order-condition-only training (e.g. an Euler residual loss)
    cannot identify.

    Parameters
    ----------
    bellman_period : BellmanPeriod
        The model Bellman Period.
    control_sym : str, optional
        Control variable symbol. Defaults to first control.
    apply_open_bounds : bool, optional
        Apply sigmoid/softplus scaling to the policy head. Default True.
    width : int, optional
        Width of hidden layers. Default 32.
    **kwargs
        Passed to :class:`Net` (activation, n_layers, init_seed, etc.).
    """

    def __init__(
        self,
        bellman_period,
        control_sym=None,
        apply_open_bounds=True,
        width=32,
        **kwargs,
    ):
        self._init_bellman_period(bellman_period, control_sym)
        self.apply_open_bounds = apply_open_bounds

        # Bounds setup (uses _setup_bound from BellmanPeriodMixin)
        self.upper_bound = self.cobj.upper_bound
        self.upper_bound_vec_func, self.upper_bound_param_to_column = self._setup_bound(
            self.upper_bound, "Upper bound"
        )
        self.lower_bound = self.cobj.lower_bound
        self.lower_bound_vec_func, self.lower_bound_param_to_column = self._setup_bound(
            self.lower_bound, "Lower bound"
        )

        # Net: shared backbone with 1 output (policy head)
        super().__init__(n_inputs=len(self.iset), n_outputs=1, width=width, **kwargs)

        # Value head: separate Linear from the shared backbone
        self.value_output = torch.nn.Linear(width, 1)
        torch.nn.init.normal_(self.value_output.weight, mean=0.0, std=0.05)
        torch.nn.init.zeros_(self.value_output.bias)
        self.value_output.to(device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        """Run shared backbone, then policy head (bounded) + value head.

        Returns the ``(policy, value)`` pair. The policy tensor is scaled
        into the control's open bounds; the value tensor is unconstrained.
        Both have shape ``(n, 1)``.
        """
        x_input = x

        # Shared hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if not self.activation_is_identity[i]:
                x = self.activations[i](x)

        # Policy head (uses Net's output layer); bounds logic shared with
        # BlockPolicyNet via BellmanPeriodMixin._apply_open_bounds.
        policy_raw = self.output(x)
        if self.transform is not None:
            policy_raw = self._apply_transform(policy_raw)
        policy = self._apply_open_bounds(policy_raw, x_input)

        # Value head (unconstrained)
        value = self.value_output(x)

        return policy, value

    # ------------------------------------------------------------------
    # Policy interface (compatible with BlockPolicyNet)
    # ------------------------------------------------------------------
    def decision_function(self, states_t, shocks_t, parameters):
        """Map states, shocks, and parameters to a controls dict.

        Parameters
        ----------
        states_t : dict
            Arrival state values, ``symbol -> tensor``.
        shocks_t : dict or None
            Shock values, ``symbol -> tensor`` (``None`` is treated as ``{}``).
        parameters : dict
            Model parameters, ``symbol -> value``.

        Returns
        -------
        dict
            ``{control_sym: tensor}`` of policy-head outputs. The arrival
            states are mapped to the control's information set via
            :meth:`~skagent.bellman.BellmanPeriod.compute_pre_state` before
            the network is evaluated.
        """
        if shocks_t is None:
            shocks_t = {}
        iset_dict = self.bellman_period.compute_pre_state(
            self.control_sym, states_t, shocks=shocks_t, parameters=parameters
        )
        iset_vals = [iset_dict[isym].flatten() for isym in self.iset]

        def get_tensor_size(d):
            for value in d.values():
                if hasattr(value, "numel"):
                    return value.numel()
                elif hasattr(value, "size"):
                    return value.size
            return 1

        dr = self.get_decision_rule(length=get_tensor_size(iset_dict))
        output = dr[self.control_sym](*iset_vals)
        return {self.control_sym: output}

    def get_decision_rule(self, length=None):
        """Decision rule returning only the policy output."""

        def decision_rule(*information):
            if len(information) > 0:
                input_tensor = torch.stack(information).T.to(device)
            else:
                if length is None:
                    raise ValueError(
                        "Must pass tensor length for empty information set in "
                        f"BlockPolicyValueNet.get_decision_rule for control '{self.control_sym}'."
                    )
                input_tensor = torch.empty(length, 0, device=device)
            policy, _value = self(input_tensor)
            return policy.flatten()

        return {self.control_sym: decision_rule}

    # ------------------------------------------------------------------
    # Value interface
    # ------------------------------------------------------------------
    def value_function(self, states_t, shocks_t=None, parameters=None):
        """Evaluate the value head at the control's information set.

        The input domain mirrors :meth:`decision_function`: arrival
        ``states_t`` (with ``shocks_t`` and ``parameters``) are mapped to the
        control's information set via
        :meth:`~skagent.bellman.BellmanPeriod.compute_pre_state`, then the
        shared backbone's value head is evaluated on that pre-decision
        representation.

        Parameters
        ----------
        states_t : dict
            Arrival state values, ``symbol -> tensor``.
        shocks_t : dict or None, optional
            Shock values (``None`` is treated as ``{}``).
        parameters : dict or None, optional
            Model parameters.

        Returns
        -------
        torch.Tensor
            Flattened value estimates, one per input row.
        """
        if shocks_t is None:
            shocks_t = {}
        iset_dict = self.bellman_period.compute_pre_state(
            self.control_sym, states_t, shocks=shocks_t, parameters=parameters
        )
        iset_vals = [iset_dict[isym].flatten() for isym in self.iset]
        input_tensor = torch.stack(iset_vals).T.to(device)
        _policy, value = self(input_tensor)
        return value.flatten()

    def get_value_function(self):
        def vf(states_t, shocks_t=None, parameters=None):
            return self.value_function(
                states_t,
                shocks_t if shocks_t is not None else {},
                parameters,
            )

        return vf

    # ------------------------------------------------------------------
    # Core function (for train_block_nn)
    # ------------------------------------------------------------------
    def get_core_function(self, length=None):
        """Return decision rules (policy head) for use with train_block_nn."""
        return self.get_decision_rule(length=length)

    def get_policy_and_value_functions(self, length=None):
        """Return both policy decision rules and value function."""
        return self.get_decision_rule(length=length), self.get_value_function()


###########
# Training Nets


# General loss function that operates on tensor and averages over samples
def aggregate_net_loss(inputs: Grid, df, loss_function):
    """
    Compute a loss function over a tensor of inputs, given a decision function df.
    Return the mean.
    """
    losses = loss_function(df, inputs)
    if not isinstance(losses, torch.Tensor):
        raise TypeError(
            "loss_function must return a torch.Tensor of per-sample losses, "
            f"got {type(losses).__name__}."
        )
    if losses.device != device:
        losses = losses.to(device)
    return losses.mean()


def train_block_nn(
    block_policy_nn,
    inputs: Grid,
    loss_function: Callable,
    epochs: int = 50,
    lr: float = 0.01,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = 1.0,
    verbose: bool = True,
):
    """Train a policy network by minimizing a loss function over a grid.

    Parameters
    ----------
    block_policy_nn : BlockPolicyNet or BlockPolicyValueNet
        The network to train. Its ``get_core_function`` supplies the
        decision rule(s) the loss is evaluated against.
    inputs : Grid
        Input grid containing states and shocks.
    loss_function : Callable
        Loss function ``(decision_function, input_grid) -> loss_tensor``.
    epochs : int, optional
        Number of training epochs (default 50).
    lr : float, optional
        Learning rate for Adam optimizer (default 0.01).
    optimizer : torch.optim.Optimizer or None, optional
        Pre-existing optimizer to reuse (preserves momentum across calls).
        If None, a new Adam optimizer is created.
    grad_clip : float or None, optional
        Maximum gradient norm for clipping (default 1.0). Set to None to disable.
    verbose : bool, optional
        Emit a ``logging.info`` message with the loss every 100 epochs
        (default True). Configure the root logger to suppress these.

    Returns
    -------
    tuple
        ``(trained_network, final_loss, optimizer)``. The ``optimizer`` is
        the one passed in, or the Adam instance created internally when none
        was supplied; returning it always lets callers warm-start a later
        call by threading it back in.
    """
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError(f"epochs must be a positive integer, got {epochs!r}")
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")
    if grad_clip is not None and grad_clip <= 0:
        raise ValueError(f"grad_clip must be > 0 or None, got {grad_clip}")

    if optimizer is None:
        optimizer = torch.optim.Adam(block_policy_nn.parameters(), lr=lr)

    # NaN sentinel (overwritten on the first epoch; epochs >= 1 is validated
    # above). Typing it as float keeps the return contract free of None.
    final_loss = float("nan")
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = aggregate_net_loss(
            inputs, block_policy_nn.get_core_function(length=inputs.n()), loss_function
        )
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(block_policy_nn.parameters(), grad_clip)
        optimizer.step()
        final_loss = loss.item()

        # final_loss is already a synced host scalar, so this guard is free.
        # A non-finite loss means gradients have diverged; continuing would
        # silently poison the weights (NaN < tol is False, so convergence
        # checks never fire). Stop and surface it instead.
        if final_loss != final_loss or final_loss in (float("inf"), float("-inf")):
            logging.warning(
                "Non-finite loss (%s) at epoch %d; stopping training early.",
                final_loss,
                epoch,
            )
            break

        if verbose and epoch % 100 == 0:
            logging.info("Epoch %d: Loss = %.6e", epoch, final_loss)

    return block_policy_nn, final_loss, optimizer
