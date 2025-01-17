from functools import wraps
import time

# import os
# import toml #not needed atm
import pkg_resources

import numpy as np


def get_version():
    return pkg_resources.get_distribution("vapc").version


# def get_version():
#     """
#     Retrieves the version of the vapc package from pyproject.toml.

#     Returns
#     -------
#     str
#         The version string of the vapc package.

#     Raises
#     ------
#     FileNotFoundError
#         If the pyproject.toml file is not found.
#     KeyError
#         If the version key is missing in the pyproject.toml file.
#     """
#     try:
#         project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#         print(project_root)
#         pyproject_path = os.path.join(project_root, 'pyproject.toml')
#         with open(pyproject_path, 'r') as f:
#             pyproject = toml.load(f)
#         return pyproject['project']['version']
#     except FileNotFoundError:
#         raise FileNotFoundError("pyproject.toml not found. Ensure it exists in the project root.")
#     except KeyError:
#         raise KeyError("Version not found in pyproject.toml. Ensure 'project.version' is specified.")

# __version__ = get_version()
# from importlib import metadata


# Read the version from package metadata
# __version__ = metadata.version(__package__)


DECORATOR_CONFIG = {"trace": True, "timeit": True}


def enable_trace(enable=True):
    """
    Enables or disables the trace decorator.

    Parameters
    ----------
    enable : bool, optional
        If True, tracing is enabled. If False, tracing is disabled.
    """
    DECORATOR_CONFIG["trace"] = enable


def enable_timeit(enable=True):
    """
    Enables or disables the timeit decorator.

    Parameters
    ----------
    enable : bool, optional
        If True, timing is enabled. If False, timing is disabled.
    """
    DECORATOR_CONFIG["timeit"] = enable


def trace(func):
    """
    A decorator that prints the name of the function before it is called.
    Controlled by the DECORATOR_CONFIG['trace'] flag.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if DECORATOR_CONFIG["trace"]:
            print(f"Calling {func.__name__}")
        return func(*args, **kwargs)

    return wrapper


def timeit(func):
    """
    A decorator that measures and prints the execution time of a function.
    Controlled by the DECORATOR_CONFIG['timeit'] flag.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if DECORATOR_CONFIG["timeit"]:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
            return result
        else:
            return func(*args, **kwargs)

    return wrapper


def compute_mode_continuous(data):
    """
    Compute the mode of a continuous dataset using a histogram.

    Parameters:
        data (array-like): The data to analyze.
    
    Returns:
        float: The mode of the dataset.
    """
    bins = optimal_bins(data, method="fd")
    # Compute the histogram and find the bin with the maximum density
    counts, bin_edges = np.histogram(data, bins=bins, density=True)
    max_bin_index = np.argmax(counts)
    # Calculate the mode as the center of the bin with the highest density
    return (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2


def optimal_bins(data, method='fd'):
    """
    Compute the optimal number of bins for a histogram using various rules.

    Parameters:
        data (array-like): The data to analyze.
        method (str): The rule to use ('sqrt', 'sturges', 'rice', 'fd', 'scott').

    Returns:
        int: The optimal number of bins.
    """
    n = len(data)
    if method == 'sqrt':
        return int(np.sqrt(n))
    elif method == 'sturges':
        return int(np.ceil(np.log2(n) + 1))
    elif method == 'rice':
        return int(np.ceil(2 * n**(1 / 3)))
    elif method == 'fd':  # Freedman-Diaconis
        iqr = np.subtract(*np.percentile(data, [75, 25]))
        bin_width = 2 * iqr / n**(1 / 3)
        return int(np.ceil((np.max(data) - np.min(data)) / bin_width))
    elif method == 'scott':
        std_dev = np.std(data)
        bin_width = 3.5 * std_dev / n**(1 / 3)
        return int(np.ceil((np.max(data) - np.min(data)) / bin_width))
    else:
        raise ValueError(f"Unknown method: {method}")
