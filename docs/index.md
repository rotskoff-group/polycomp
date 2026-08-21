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

For those interested in the underlying methods, the Conceptual Basis section provides a good primer, alongside the references set within. 
This project is indebted to the extensive work of those who first developed field theory methods, and we hope to help enable future researchers to push them further. 
