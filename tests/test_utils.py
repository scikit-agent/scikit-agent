import skagent.utils as utils


def test_utils_apply_fun_to_vals():
    def pow(x, y):
        return x**y

    vals = {"x": 2, "y": 3}
    a = utils.apply_fun_to_vals(pow, vals)
    assert a == 8
