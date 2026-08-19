#polycomp

Python implementation of polymer auxiliary field theory with complex Langevin integation (CL-FTS).
This implementation is designed for solving fully-fluctuating polymer field theories with the 
auxiliary field theory method, and is particularly well suited for charged polymers and coacervation
behavior. Can be used to compute density profiles and other major observables for 1D, 2D and 3D polymer
melts. Currently restricted to linear polymers. 

Our current development has placed an emphasis on simulations 
of charged linear polymers, but the platform is flexible and we plan to continue 
development on many systems. The code runs only on GPUs due to the computational cost
of the simulations, and has been optimized for those systems. 

For those interested in the underlying methods, the attached references are the best technical description
of the underlying equations that this code is designed to solve. The first two references provide direct
explanations of the implementation present in this codebase, while the third is the primary general reference 
work for field theory methods in general and was indispensable in creating the codebase, though not directly
affiliated with this project. 

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
