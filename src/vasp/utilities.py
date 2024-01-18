from functools import wraps
import time

def trace(func):
    @wraps(func)
    def call(*args,**kwargs):
        print("Calling",func.__name__)
        return func(*args,**kwargs)
    return call

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Capture the start time
        result = func(*args, **kwargs)  # Call the function
        end_time = time.time()  # Capture the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper