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

Polymer field theories are a powerful set of simulation tools for determining phase and
structural information about polymer assemblies. 
Polymer field theories represent the structure of polymer assemblies using fields for 
polymer density and chemical potential, allowing for efficient computations that scale
well to large systems because they do not require computing pairwise interactions. 
The methods are well-validated for a wide variety of systems, but open source code is 
not commonly available to compute systems that cannot be captured with self-consistent
methods. 
The class of broader systems that require including fluctuations includes most charged 
polymers, particularly bio-polymers, which this work aims to address. 

# Statement of need

`PolyComp` is a python-based implementation of Field Theoretic Simulation 
for an Auxiliary Field Theory with Complex Langevin integration. 
It is optimized for and runs exclusively on GPUs using CuPy to handle lower level 
operations while still being readable to any user familiar with NumPy. 
The API is designed to allow for straightforward simulation of linear polymer melts and
can also accommodate nanoparticles and affixed polymer brushes. 
The development of the underlying methods has been ongoing for many years and our 
implementation provides small modifications to the common polymer field theory methods
[@fredricksonEquilibriumTheoryInhomogeneous2006]
[@fredricksonFieldTheoreticSimulationsSoft2023], mainly around the handling of the 
systems incompressibility constraint. 
The code uses efficient pseudospectral integrators to compute the most expensive 
portions of the underlying theory. 

`PolyComp` has been used to simulate biological systems examining mRNA encapsulation 
for drug delivery [@pertCoacervationDrivesMorphological2025], formation of nanoparticle
superlattices [@yeNanoparticleSuperlatticesAssembled2025] and as a base for the
development of machine learned acceleration to general polymer field theories
[@pertScalingFieldTheoreticSimulation2025]. 
The code is written completely in python, and should allow easy implementation of 
performant polymer simulation for anyone familiar to the language. 
The source code for PolyComp has been
archived to Zenodo with the linked DOI: [@polycomp_zenodo]

# Acknowledgements

We acknowledge contributions from the many members of the Rotskoff Group who have helped
with conceptualization, debugging, and optimization of this project, in particular 
Sherry Li, Clay Batton, Andy Mitchell, and Nicholas Juntunen. 

# References

