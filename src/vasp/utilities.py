from functools import wraps
import time

def trace(func):
    """
    A decorator that prints the name of the function before it is called.

    This decorator is useful for debugging and logging purposes, allowing you to trace
    the flow of function calls in your application by outputting "Calling <function_name>"
    to the console before the function executes.

    Args:
        func (callable): The function to be wrapped.

    Returns:
        callable: The wrapped function that prints its name before execution.

    Example:
        >>> @trace
        ... def greet(name):
        ...     print(f"Hello, {name}!")
        >>> greet("Alice")
        Calling greet
        Hello, Alice!
    """
    @wraps(func)
    def call(*args,**kwargs):
        print("Calling",func.__name__)
        return func(*args,**kwargs)
    return call

def timeit(func):
    """
    A decorator that measures and prints the execution time of a function.

    This decorator records the start and end times of the function execution using `time.time()`,
    calculates the elapsed time, and prints it in seconds with four decimal places of precision.
    It's useful for performance testing or profiling to see how long functions take to execute.

    Args:
        func (callable): The function to be wrapped.

    Returns:
        callable: The wrapped function that measures and prints its execution time.

    Example:
        >>> @timeit
        ... def compute():
        ...     time.sleep(1)
        >>> compute()
        Function 'compute' executed in 1.0001 seconds

    Notes:
        - The timing is based on wall-clock time, which can be affected by other system processes.
        - For more precise timing (e.g., CPU time), consider using `time.perf_counter()`.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Capture the start time
        result = func(*args, **kwargs)  # Call the function
        end_time = time.time()  # Capture the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper