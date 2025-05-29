import skagent.algos.maliar as solver
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

        input_tensor = torch.stack(iset_vals).T

        input_tensor = input_tensor.to(device)
        output = self(input_tensor)  # application of network

        # again, assuming only one for now...
        # decisions = dict(zip([csym], output))
        decisions = {csym: output}
        return decisions

    def get_decision_function(self):
        def df(states_t, shocks_t, parameters):
            return self.decision_function(states_t, shocks_t, parameters)

        return df


################
# Model bindings


def get_estimated_discounted_lifetime_reward_loss(
    state_variables, block, discount_factor, big_t, parameters
):
    # TODO: Should be able to get 'state variables' from block
    # Maybe with ZP's analysis modules

    # convoluted
    # TODO: codify this encoding and decoding of the grid into a separate object
    shock_vars = block.get_shocks()
    big_t_shock_syms = sum(
        [[f"{sym}_{t}" for sym in list(shock_vars.keys())] for t in range(big_t)], []
    )

    # will work for big_t = 1 only.
    given_syms = state_variables + big_t_shock_syms

    def estimated_discounted_lifetime_reward_loss(df, input_vector):
        ## includes the values of state_0 variables, and shocks.
        given_vals = dict(zip(given_syms, input_vector))

        shock_vals = {sym: given_vals[sym] for sym in big_t_shock_syms}
        shocks_by_t = {
            sym: torch.stack([shock_vals[f"{sym}_{t}"] for t in range(big_t)])
            for sym in shock_vars
        }

        ####block, discount_factor, dr, states_0, big_t, parameters={}, agent=None
        edlr = solver.estimate_discounted_lifetime_reward(
            block,
            discount_factor,
            df,
            {sym: given_vals[sym] for sym in state_variables},
            big_t,
            parameters=parameters,
            agent=None,  ## TODO: Pass through the agent?
            shocks_by_t=shocks_by_t,
            ## Handle multiple decision rules?
        )
        return -edlr

    return estimated_discounted_lifetime_reward_loss


###########
# Training Nets


# General loss function that operates on tensor and averages over samples
def aggregate_net_loss(inputs, df, loss_function):
    """
    Compute a loss function over a tensor of inputs, given a decision function df.
    Return the mean.
    """
    # we include the network as a potential input to the loss function
    losses = torch.stack(
        [loss_function(df, inputs[i, :]) for i in range(inputs.shape[0])]
    )
    losses = losses.to(device)
    return losses.mean()


def train_block_policy_nn(block_policy_nn, inputs, loss_function, epochs=50):
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
