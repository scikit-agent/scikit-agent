import inspect
from skagent.grid import Grid
import torch
from skagent.utils import create_vectorized_function_wrapper_with_mapping

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
    """
    Parameters
    -----------

    apply_open_bounds: boolean
        If True, then the network forward output is normalized by the upper and/or lower bounds,
        computed as a function of the input tensor. These bounds are "open" because output
        can be arbitrarily close to, but not equal to, the bounds.
    """

    def __init__(self, block, width=32, apply_open_bounds=True):
        self.block = block
        self.apply_open_bounds = apply_open_bounds

        ## pseudo -- assume only one for now
        # assuming only on control for now
        self.csym = self.block.get_controls()[0]
        self.control = self.block.dynamics[self.csym]
        self.iset = self.control.iset

        ## assess whether/how the control is bounded
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

        super().__init__(
            len(self.iset),
            1,
            width,
        )

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
            if self.upper_bound and self.lower_bound:
                # Compute bounds from input using wrapped functions
                upper_bound = self.upper_bound_vec_func(x)
                lower_bound = self.lower_bound_vec_func(x)

                # Scale to bounds
                x2 = lower_bound + torch.nn.functional.sigmoid(x1) * (
                    upper_bound - lower_bound
                )

            if self.lower_bound and not self.upper_bound:
                lower_bound = self.lower_bound_vec_func(x)
                x2 = lower_bound + torch.nn.functional.softplus(x1)

            if not self.lower_bound and self.upper_bound:
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
