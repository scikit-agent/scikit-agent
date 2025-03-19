import skagent.utils as utils


def test_utils_apply_fun_to_vals():
    fun = lambda x,y: x ** y
    vals = {
        'x' : 2,
        'y' : 3
    }
    a = utils.apply_fun_to_vals(fun, vals)
    assert(a == 8)