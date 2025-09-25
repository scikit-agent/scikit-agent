import skagent.utils as utils
import torch


def test_utils_apply_fun_to_vals():
    def pow(x, y):
        return x**y

    vals = {"x": 2, "y": 3}
    a = utils.apply_fun_to_vals(pow, vals)
    assert a == 8


def test_extract_parameters():
    """Test parameter extraction from neural networks."""
    # Create a simple network
    net = torch.nn.Sequential(
        torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 1)
    )

    # Extract parameters
    params = utils.extract_parameters(net)

    # Should be a 1D tensor with all parameters flattened
    assert isinstance(params, torch.Tensor)
    assert params.dim() == 1

    # Expected number of parameters: (2*3 + 3) + (3*1 + 1) = 9 + 4 = 13
    expected_params = (2 * 3 + 3) + (3 * 1 + 1)
    assert params.shape[0] == expected_params


def test_compute_parameter_difference():
    """Test parameter difference computation."""
    # Create two identical networks
    net1 = torch.nn.Linear(2, 1)
    net2 = torch.nn.Linear(2, 1)

    # Copy parameters from net1 to net2
    net2.load_state_dict(net1.state_dict())

    params1 = utils.extract_parameters(net1)
    params2 = utils.extract_parameters(net2)

    # Should have zero difference
    diff = utils.compute_parameter_difference(params1, params2)
    assert abs(diff) < 1e-6

    # Modify net2 slightly
    with torch.no_grad():
        net2.weight += 0.1

    params2_modified = utils.extract_parameters(net2)
    diff_modified = utils.compute_parameter_difference(params1, params2_modified)

    # Should have non-zero difference
    assert diff_modified > 0
