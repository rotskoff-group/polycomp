# Getting Started

## Installation

As the code is still under active development, it should be installed by downloading 
from our [github](https://github.com/rotskoff-group/polycomp) and installing locally via 

    pip install .

## Managing dependencies

You can find all of the required packages prepared in a single conda environment which
can be installed via

    conda env create --file polycomp.yml

The package will also require a working CUDA installation as polycomp only runs on GPUs

## Running the tests

You can make sure that your installation has worked by running 
    
    cd tests
    python tests.py

from the main directory directory. 

You can also try try running some of the [examples](examples.md). 
