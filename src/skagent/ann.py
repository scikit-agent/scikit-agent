import inspect
import logging
import math
from skagent.block import normalize_bound
from skagent.grid import Grid
import torch
from skagent.utils import (
    create_vectorized_function_wrapper_with_mapping,
    require_positive_integer,
)
from typing import Callable, Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hidden-layer activations a Net accepts by name. The identity maps to None,
# which the forward pass skips.
_ACTIVATIONS = {
    "silu": torch.nn.functional.silu,
    "relu": torch.nn.functional.relu,
    "tanh": torch.nn.functional.tanh,
    "sigmoid": torch.nn.functional.sigmoid,
    "identity": None,
}

# Output transforms a Net accepts by name; see _apply_output_transform.
_OUTPUT_TRANSFORMS = {
    "sigmoid": torch.sigmoid,
    "exp": torch.exp,
    "tanh": torch.tanh,
    "relu": torch.nn.functional.relu,
    "softplus": torch.nn.functional.softplus,
    "softmax": lambda x: torch.nn.functional.softmax(x, dim=-1),
    "abs": torch.abs,
    "square": lambda x: x**2,
    "identity": lambda x: x,
}


def _activation_fn(activation):
    """Resolve an activation name, callable, or None; None means the identity."""
    if activation is None:
        return None
    if isinstance(activation, str) and activation in _ACTIVATIONS:
        return _ACTIVATIONS[activation]
    if callable(activation):
        return activation
    raise ValueError(f"Unsupported activation: {activation}")


def _apply_output_transform(x, transform):
    """Apply one output transform (a name, a callable, or None) to *x*."""
    if transform is None:
        return x
    if isinstance(transform, str) and transform in _OUTPUT_TRANSFORMS:
        return _OUTPUT_TRANSFORMS[transform](x)
    if callable(transform):
        return transform(x)
    raise ValueError(f"Unknown single transform: {transform}")


def _batch_length(*dicts):
    """Number of samples in the first tensor or array found in *dicts*, else 1."""
    for d in dicts:
        for value in d.values():
            if hasattr(value, "numel"):
                return value.numel()
            if hasattr(value, "size"):
                return value.size
    return 1


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
        that :meth:`get_decision_function` returns.
        """
        raise NotImplementedError

    def _network_input(self, states_t, shocks_t, parameters):
        """Map arrival states to the ``(n, len(iset))`` network input.

        The control's information set is computed with
        :meth:`~skagent.bellman.BellmanPeriod.compute_pre_state`. When it is
        empty the network is a constant, and ``n`` comes from the states.
        """
        iset_dict = self.bellman_period.compute_pre_state(
            self.control_sym, states_t, shocks=shocks_t, parameters=parameters
        )
        return self._stack_information(
            [iset_dict[isym].flatten() for isym in self.iset],
            _batch_length(states_t, shocks_t or {}),
        )

    def _stack_information(self, information, length):
        """Stack information-set columns into an ``(n, k)`` input on *device*.

        An empty information set gives an ``(length, 0)`` input, so *length*
        is required then.
        """
        if len(information) > 0:
            return torch.stack(information).T.to(device)
        if length is None:
            raise ValueError(
                "Must pass a tensor length for an empty information set "
                f"(control '{self.control_sym}')."
            )
        return torch.empty(length, 0, device=device)

    def _setup_bounds(self):
        """Vectorize the control's upper and lower bounds over the iset columns."""
        self.upper_bound = self.cobj.upper_bound
        self.upper_bound_vec_func, self.upper_bound_param_to_column = self._setup_bound(
            self.upper_bound, "Upper bound"
        )
        self.lower_bound = self.cobj.lower_bound
        self.lower_bound_vec_func, self.lower_bound_param_to_column = self._setup_bound(
            self.lower_bound, "Lower bound"
        )

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
        """Set up a vectorized bound function from a callable, number, or None.

        The bound is normalized with :func:`skagent.block.normalize_bound`: a
        number becomes a zero-argument callable, a callable is used as-is, and
        ``None`` disables the bound on that side. ``Control`` already
        normalizes its bounds at construction, so this is normally a no-op;
        it also covers bounds set directly on the control object.
        """
        bound_func = normalize_bound(bound_func, bound_name)
        if bound_func is None:
            return None, None
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
        """Return a callable ``(states, shocks, parameters) -> controls dict``."""
        return self.decision_function


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
        # copy_weights_from is used below but never stored: assigning a Module
        # to an attribute registers it as a submodule, and its parameters
        # would then count as this network's.

        self._set_activations(activation, n_layers)

        # Build network layers
        self.layers = torch.nn.ModuleList()

        # First hidden layer
        self.layers.append(torch.nn.Linear(n_inputs, width))

        # Additional hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(torch.nn.Linear(width, width))

        # Output layer, plus any heads a subclass adds before the weights are
        # drawn, so a seeded initialisation covers them too
        self.output = torch.nn.Linear(width, n_outputs)
        self._add_heads(width)

        # Initialize weights first (before device placement)
        self._init_weights(init_seed)

        # Copy weights AFTER initialization if requested
        if copy_weights_from is not None:
            self._copy_weights_from_network(copy_weights_from)

        # Move to device for backward compatibility (after all initialization)
        self.to(device)

    def _add_heads(self, width):
        """Add output layers beyond ``self.output``; none by default."""

    def _set_activations(self, activation, n_layers):
        """Set the per-layer activations and record which are the identity."""
        if not isinstance(activation, list):
            # Single activation applied to all layers
            activation = [activation] * n_layers
        if len(activation) != n_layers:
            raise ValueError(
                f"Number of activations ({len(activation)}) must match "
                f"number of layers ({n_layers})"
            )
        self.activations = [_activation_fn(act) for act in activation]
        self.activation_is_identity = [fn is None for fn in self.activations]

    def _init_weights(self, init_seed):
        """Draw the initial weights, under *init_seed* if one is given.

        The global RNG state is restored afterwards, so seeding one network's
        initialisation leaves every later draw where it would have been.
        """
        if init_seed is not None:
            current_state = torch.get_rng_state()
            torch.manual_seed(init_seed)

        # Custom weight initialisation to match MMW notebook (normal std=0.05)
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.05)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        if init_seed is not None:
            torch.set_rng_state(current_state)

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

    @property
    def device(self):
        """Device property for backward compatibility."""
        return next(self.parameters()).device

    def _hidden(self, x):
        """Run the hidden layers, skipping identity activations."""
        for layer, activation in zip(self.layers, self.activations):
            x = layer(x)
            if activation is not None:
                x = activation(x)
        return x

    def _output_head(self, h):
        """Apply the output layer and any output transform to hidden state *h*."""
        return self._apply_transform(self.output(h))

    def forward(self, x):
        return self._output_head(self._hidden(x))

    def _apply_transform(self, x):
        """Apply output transformation based on configuration."""
        if not isinstance(self.transform, list):
            # Single transform (or None) applied to all outputs
            return _apply_output_transform(x, self.transform)

        # List of transforms: apply each transform to corresponding output
        if len(self.transform) != x.shape[-1]:
            raise ValueError(
                f"Number of transforms ({len(self.transform)}) must match "
                f"number of outputs ({x.shape[-1]})"
            )
        return torch.stack(
            [
                _apply_output_transform(x[..., i], transform)
                for i, transform in enumerate(self.transform)
            ],
            dim=-1,
        )


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

        # Vectorize the control's bounds over the information-set columns;
        # the forward pass scales the output into them.
        self._setup_bounds()

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
        x = self._network_input(states_t, shocks_t, parameters)
        return {self.control_sym: self._policy(x).flatten()}

    def _bounded_policy(self, h, x):
        """Policy output from hidden state *h*, scaled into the bounds at input *x*."""
        return self._apply_open_bounds(self._output_head(h), x)

    def _policy(self, x):
        """The policy at network input *x*; subclasses with more heads select it."""
        return self(x)

    def forward(self, x):
        """
        Note that this uses the same architecture of the superclass
        but adds on a normalization layer appropriate to the
        bounds of the decision rule.
        """
        return self._bounded_policy(self._hidden(x), x)

    def get_core_function(self, length=None):
        return self.get_decision_rule(length=length)

    def get_decision_rule(self, length=None):
        """
        Returns the decision rule corresponding to this neural network.

        The rule takes the information-set values as positional arguments and
        returns the control values. For an empty information set, *length* is
        the number of samples to return.
        """

        def decision_rule(*information):
            x = self._stack_information(information, length)
            return self._policy(x).flatten()

        return {self.control_sym: decision_rule}


class BlockValueNet(BellmanPeriodMixin, Net):
    """Standalone value-function network for a Bellman problem.

    Maps a control's information set (the same pre-decision states a policy
    network sees) to a single unconstrained scalar value. It is the value-only
    counterpart of :class:`BlockPolicyNet`, kept for algorithms that approximate
    a value function separately from the policy. The Maliar/MMW path in this
    package uses the shared-backbone :class:`BlockPolicyValueNet` instead, so
    ``BlockValueNet`` is not wired into
    :func:`~skagent.algos.maliar.maliar_training_loop`.

    Parameters
    ----------
    bellman_period : BellmanPeriod
        The model Bellman period.
    control_sym : str, optional
        Control whose information set defines the value function's domain.
        Defaults to the first control.
    width : int, optional
        Width of hidden layers. Default 32.
    **kwargs
        Passed to :class:`Net` (activation, n_layers, init_seed, etc.).
    """

    def __init__(self, bellman_period, control_sym=None, width: int = 32, **kwargs):
        self._init_bellman_period(bellman_period, control_sym)
        # Information set in, single unconstrained scalar value out.
        super().__init__(n_inputs=len(self.iset), n_outputs=1, width=width, **kwargs)

    def value_function(self, states_t, shocks_t=None, parameters=None):
        """Evaluate the value function at the control's information set.

        Arrival ``states_t`` (with ``shocks_t`` and ``parameters``) are mapped to
        the control's information set via
        :meth:`~skagent.bellman.BellmanPeriod.compute_pre_state`, mirroring
        :meth:`BlockPolicyNet.decision_function`, then the network is evaluated.

        Returns
        -------
        torch.Tensor
            Flattened value estimates, one per input row.
        """
        return self(self._network_input(states_t, shocks_t, parameters)).flatten()

    def get_value_function(self):
        """Return a callable ``(states, shocks, parameters) -> value`` tensor."""
        return self.value_function

    def get_core_function(self, length=None):
        """Return the value function (the trainable core for this net)."""
        return self.get_value_function()


class BlockPolicyValueNet(BlockPolicyNet):
    """
    Single neural network with shared backbone for both policy and value.

    A :class:`BlockPolicyNet` with a second, unconstrained output head; the
    policy interface (decision function, decision rule, core function) is
    inherited unchanged and reads the policy head.

    Architecture: shared hidden layers → two output heads:
    - **Policy head**: bounded output (sigmoid-scaled to satisfy constraints)
    - **Value head**: unconstrained scalar output

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

    def _add_heads(self, width):
        # Value head: a second Linear on the shared backbone, drawn with the
        # rest of the weights so init_seed fixes it too
        self.value_output = torch.nn.Linear(width, 1)

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        """Run shared backbone, then policy head (bounded) + value head.

        Returns the ``(policy, value)`` pair. The policy tensor is scaled
        into the control's open bounds; the value tensor is unconstrained.
        Both have shape ``(n, 1)``.
        """
        h = self._hidden(x)
        return self._bounded_policy(h, x), self.value_output(h)

    def _policy(self, x):
        return self(x)[0]

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
        x = self._network_input(states_t, shocks_t, parameters)
        return self(x)[1].flatten()

    def get_value_function(self):
        """Return a callable ``(states, shocks, parameters) -> value`` tensor."""
        return self.value_function

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
    return losses.to(device).mean()


def _validate_train_args(epochs, lr, grad_clip):
    """Validate the scalar arguments of :func:`train_block_nn`."""
    require_positive_integer("epochs", epochs)
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")
    if grad_clip is not None and grad_clip <= 0:
        raise ValueError(f"grad_clip must be > 0 or None, got {grad_clip}")


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

    This is a generic stochastic-gradient-descent driver, not a solution
    algorithm in itself. It runs ``epochs`` Adam updates that minimize whatever
    ``loss_function`` is supplied, evaluated on a single, *fixed* grid of
    ``inputs``; it is agnostic to where that grid came from or which method the
    loss encodes (Euler residual, Bellman residual, FOC, or a custom loss).

    Because it trains on whatever ``inputs`` it is given, accuracy depends on
    the caller re-sampling those states across calls: Maliar, Maliar, and Winant
    (2021) keep the training data "constantly re-sampled," and minimizing on a
    single fixed grid instead lets the solution over-fit those points while
    drifting elsewhere. Re-draw ``inputs`` each call (threading the returned
    optimizer back in to keep Adam's momentum), or use
    :func:`~skagent.algos.maliar.maliar_training_loop`, which wraps this driver
    in the full MMW'21 outer loop: it alternates these inner SGD updates with a
    forward-simulation step that refreshes the training states toward the
    model's ergodic set.

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
        Number of training epochs (default 50). Any integer type is accepted,
        numpy integers included.
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
    _validate_train_args(epochs, lr, grad_clip)

    if optimizer is None:
        optimizer = torch.optim.Adam(block_policy_nn.parameters(), lr=lr)

    # NaN sentinel (overwritten on the first epoch; epochs >= 1 is validated
    # above). Typing it as float keeps the return contract free of None.
    final_loss = float("nan")
    # The core function reads the network's current weights on every call, so
    # one built before the loop serves every epoch.
    core_function = block_policy_nn.get_core_function(length=inputs.n())
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = aggregate_net_loss(inputs, core_function, loss_function)
        # Check finiteness BEFORE backward/step: a non-finite loss means
        # training diverged; stopping here keeps the last finite weights instead
        # of applying NaN/Inf gradients. final_loss is already synced (free).
        final_loss = loss.item()
        if not math.isfinite(final_loss):
            logging.warning(
                "Non-finite loss (%s) at epoch %d; stopping training early.",
                final_loss,
                epoch,
            )
            break

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(block_policy_nn.parameters(), grad_clip)
        optimizer.step()

        if verbose and epoch % 100 == 0:
            logging.info("Epoch %d: Loss = %.6e", epoch, final_loss)

    return block_policy_nn, final_loss, optimizer
