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
    def __init__(self, block, csym=None, width=32):
        self.block = block

        ## pseudo -- assume only one for now
        if csym is None:
            csym = next(iter(self.block.get_controls()))

        self.csym = csym
        self.cobj = self.block.dynamics[csym]

        super().__init__(len(self.cobj.iset), 1, width)

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

        iset_vals = [post[isym].flatten() for isym in self.cobj.iset]

        output = self.get_decision_rule(length=next(iter(post.values())).numel())[
            self.csym
        ](*iset_vals)

        # again, assuming only one for now...
        # decisions = dict(zip([csym], output))
        # ... when using multiple csyms, note the orientation of the output tensor
        decisions = {self.csym: output}
        return decisions

    def get_decision_function(self):
        def df(states_t, shocks_t, parameters):
            return self.decision_function(states_t, shocks_t, parameters)

        return df

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

        return {self.csym: decision_rule}


class BlockValueNet(Net):
    """
    A neural network for approximating value functions in dynamic programming problems.

    This network takes state variables as input and outputs value estimates.
    It's designed to work with the Bellman equation loss functions in the Maliar method.
    """

    def __init__(self, block, csym=None, width: int = 32):
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
        if csym is None:
            csym = next(iter(self.block.get_controls()))

        self.csym = csym
        self.cobj = self.block.dynamics[csym]

        # Use the same information set as the policy network
        self.state_variables = sorted(list(self.cobj.iset))

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
        # The inputs to the network are the information set variables
        # Combine states_t and shocks_t to get all available variables
        all_vars = states_t | shocks_t
        iset_vals = [all_vars[isym].flatten() for isym in self.cobj.iset]

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
            inputs, block_policy_nn.get_decision_rule(length=inputs.n()), loss_function
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
