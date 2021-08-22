
from inspect import signature

def filter_args(func, keys):
    """
    Filters a dictionary so it only contains keys that are arguments of a function

    Parameters
    ----------
    func : Function
        Function for which we are filtering the dictionary
    keys : dict
        Dictionary with keys we are filtering

    Returns
    -------
    filtered : dict
        Dictionary containing only keys that are arguments of func
    """
    filtered = {}
    sign = list(signature(func).parameters.keys())
    for k, v in {**keys}.items():
        if k in sign:
            filtered[k] = v
    return filtered