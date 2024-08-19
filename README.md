# Polycomp 

Python implmentation of polymer field theories with complex Langevin integration for 
determining structure of polymer melts. Particular focus on polyelectrolytes. Still 
under construction. 


install with 

    pip install . 

You can find all of the required packages prepared in a single conda environment which can be installed via

    conda env create --file polycomp.yml
The package will also require a working CUDA installation.

Running the tests
You can make sure that your installation has worked by running

    python tests.py
from the tests directory.

[Full documentation](https://rotskoff-group.github.io/polycomp) and will continue to be udated as we improve the code


