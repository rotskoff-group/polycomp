---
title: 'PolyComp: open source field theory for polymers'
tags:
  - Python
  - polymers
  - statistical mechanics
  - simulation
  - biophysics
authors:
  - name: Emmit Pert
    orcid: 0009-0005-8763-3226
    affiliation: 1 
  - name: Grant Rotskoff
    corresponding: true
    affiliation: 1
affiliations:
 - name: Stanford University
   index: 1
date: 3 Nov 2025
bibliography: paper.bib

---

# Summary

Field theoretic methods provide insight into the phase behavior of complex polymer 
assemblies, but sophisticated numerical methods are required to efficiently simulate 
nontrivial systems. 
By representing the constituent polymers through their density and chemical potential 
fields, the mesoscale organization can be determined without explicitly modeling the 
microscopic degrees of freedom, which leads to superior scalability compared with 
particle-based simulations. 
Polymer field theories accurately model a wide variety of systems, but currently 
open-source codes that implement state-of-the-art methods for field theoretic simulation
are not broadly available. This software provides a flexible platform for numerical 
simulations of polymer field theories, focused on applications to charged polymers.

# Statement of need

`PolyComp` is a Python-based implementation of Field Theoretic Simulation 
for an Auxiliary Field Theory with Complex Langevin integration. 
It is optimized for and runs on GPU hardware using CuPy to handle lower-level 
operations while still being readable to any user familiar with NumPy. 
The API is designed to allow for straightforward simulation of linear polymer melts and
can also accommodate nanoparticles and affixed polymer brushes. 
The development of the underlying methods has been ongoing for many years and our 
implementation provides small modifications to the common polymer field theory methods
[@fredricksonEquilibriumTheoryInhomogeneous2006]
[@fredricksonFieldTheoreticSimulationsSoft2023], mainly around the handling of the 
system's incompressibility constraint. 
Although the field is well-developed, the lack of open source code was a significant 
impediment to our work, and this release of a simpler, Python-based package will
save significant time for those entering the field and looking to apply these methods.
The source code for PolyComp has been
archived to Zenodo with the linked DOI: [@polycomp_zenodo].
Detailed derivations of methods are available in a previous publication 
[@pertCoacervationDrivesMorphological2025].

# Software Design

`PolyComp`'s main differentiating feature upon development was its open source nature. 
Developed concurrently, the only other fully open-source codebase we are aware of is 
`langevin-fts` [@yongDynamicProgrammingChain2025], and we developed our own 
open-source codebase to fill a gap in the literature for open implementations of 
the techniques described by existing field theory methods 
[@fredricksonEquilibriumTheoryInhomogeneous2006]. 
The design principles for the codebase
were transparency, functionality, and efficiency. 
Because the math motivating field theory simulations is quite dense, the code is written
to be as transparent as possible, allowing for easy comprehension, modification, and 
verification. 
In particular, the usage of `CuPy` for the bulk of the operations means that the code 
is parseable by any programmer who is familiar with the common `NumPy` library. 
Secondly, the code is functional, with a particular emphasis on interfacing with complex 
biological systems and machine learning methods for optimization. 
It has been extensively tested for correctness, and the native Python makes interfaces 
with modern machine learning methods for optimization easy. 
Finally, the code is efficient, having been optimized for GPU performance with both 
`CuPy`'s FFT library and our own kernels to speed up the expensive operation of solving 
the modified diffusion equation. 

# Research impact statement

`PolyComp` has been used to simulate biological systems examining mRNA encapsulation 
for drug delivery [@pertCoacervationDrivesMorphological2025], formation of nanoparticle
superlattices [@yeNanoparticleSuperlatticesAssembled2025] and as a base for the
development of machine-learned acceleration to general polymer field theories
[@pertScalingFieldTheoreticSimulation2025]. 
The code is well-adapted for biological 
simulation and is currently being used to simulate intrinsically disordered proteins
in our group. 
While the usage of the code thus far has been limited to our group, it is becoming a 
workhorse program internally and its simple usage, documentation, and proven 
usefulness means that we hope it will be adopted more broadly for those looking to 
start learning about polymer field theory simulation or running experiments with 
polymeric systems. 

# AI usage disclosure
The core functionality of the code was primarily developed from 2020-2022, without the
use of any AI assistance. Later Google Gemini was used to help develop unit tests, 
proofread code, and build documentation to prepare for the full release. 

# Acknowledgements

We acknowledge contributions from the many members of the Rotskoff Group who have helped
with conceptualization, debugging, and optimization of this project, in particular 
Sherry Li, Clay Batton, Andy Mitchell, and Nicholas Juntunen. 

# References

