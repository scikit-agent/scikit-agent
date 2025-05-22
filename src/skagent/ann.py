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


class PolicyNet(Net):
    def __init__(self, state_vars, control_vars, width=32):
        super().__init__(len(state_vars), len(control_vars), width)


################
# Model bindings


def net_to_decision_function(net, state_variables, control_variables):
    def decision_function(states_t, parameters={}):
        # could do more to sort the values of states_t by states_variables
        # because they might be unaligned
        input_tensor = torch.FloatTensor(list(states_t.values()))
        input_tensor = input_tensor.to(device)
        output = net(input_tensor)
        decisions = dict(zip(control_variables, output))
        return decisions

    return decision_function


def get_estimated_discounted_lifetime_reward_loss(
    state_variables, block, discount_factor, big_t, parameters
):
    def estimated_discounted_lifetime_reward_loss(net, input_vector):
        ### need to zip the state variables and the output vector
        states_0 = dict(zip(state_variables, input_vector))

        ####block, discount_factor, dr, states_0, big_t, parameters={}, agent=None
        edlr = solver.estimate_discounted_lifetime_reward(
            block,
            discount_factor,
            net_to_decision_function(
                net, state_variables, block.get_controls()
            ),  # TODO: Get controls by agent?
            states_0,
            big_t,
            parameters=parameters,
            agent=None,  ## TODO: Pass through the agent?
            ## Handle multiple decision rules?
        )
        return -edlr

    return estimated_discounted_lifetime_reward_loss


###########
# Training Nets


# General loss function that operates on tensor and averages over samples
def aggregate_net_loss(inputs, net, loss_function):
    # we include the network as a potential input to the loss function
    losses = torch.stack(
        [loss_function(net, inputs[i, :]) for i in range(inputs.shape[0])]
    )
    losses = losses.to(device)
    return losses.mean()


def train_nn(net, inputs, loss_function, epochs=50):
    ## to change
    # criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)  # Using Adam

    for epoch in range(epochs):
        running_loss = 0.0
        optimizer.zero_grad()
        loss = aggregate_net_loss(inputs, net, loss_function)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if epoch % 100 == 0:
            print("Epoch {}: Loss = {}".format(epoch, loss.cpu().detach().numpy()))

    return net
