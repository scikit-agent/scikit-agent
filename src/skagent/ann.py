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
