# script.py
from line_profiler import LineProfiler

lp = LineProfiler()

def inner_function():
    # Code for inner function
    pass

def outer_function():
    # Code for outer function
    inner_function()

if __name__ == "__main__":
    lp.add_function(inner_function)
    lp.add_function(outer_function)
    
    lp.run('outer_function()')
    lp.print_stats()

