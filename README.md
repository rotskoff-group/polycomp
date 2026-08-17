# Polycomp 

Python implementation of polymer field theories with complex Langevin integration for 
determining structure of polymer melts. Particular focus on polyelectrolytes. Still 
under construction. 

[Full documentation](https://rotskoff-group.github.io/polycomp) is available and will continue to be updated as we improve the code

The code requires a CUDA-compatible GPU and does not include a CPU mode. 

## Installation

We recommend using Conda to manage your environment, as it will automatically resolve the correct CUDA toolkit for your hardware.

```bash
conda env create --file polycomp.yml
conda activate polycomp
pip install .
```

Alternatively, the code and required dependencies can be installed purely via pip if you already 
have a working CUDA environment. If you do so please ensure your host CUDA version supports your 
hardware and matches your installed CuPy wheel (e.g., `cupy-cuda12x`).

```bash
pip install -r requirements.txt
pip install .
```
## Running Tests   

You can verify your installation by running the following from the current directory.

```bash     
cd test
python tests.py
```

You can also try out some of the examples, all of which should run with a successful installation. 

## Contributing and Support

We welcome contributions, bug reports, and questions! Please review our [Contributing Guidelines](CONTRIBUTING.md) for instructions on how to submit code, report issues, or seek support. If you encounter a problem, please [open an issue](https://github.com/rotskoff-group/polycomp/issues) on GitHub.

## References & Further Reading

* **Primary Methodological Reference:** 
  Pert, E. K., Hurst, P. J., Waymouth, R. M., & Rotskoff, G. M. (2025). *"Coacervation drives morphological diversity of mRNA encapsulating nanoparticles"*, The Journal of Chemical Physics, 162(7), 074902. https://doi.org/10.1063/5.0235799
  *(Full mathematical description of main underlying methods).*

* **Open-Access Technical Derivation:** 
  Pert, E. K. (2025). *"Polymer Field Theories for Biological Condensates"*, PhD Thesis, Stanford University. https://searchworks.stanford.edu/view/in00000866764
  *(Open-source thesis with full description of underlying methods, including nanoparticles).*

* **Foundational Literature:** 
  Fredrickson, G. H. (2006). *The Equilibrium Theory of Inhomogeneous Polymers*. Oxford University Press.
  *(Full description of original derivation of underlying methods, best refernce for general information about polymer field theories).*
