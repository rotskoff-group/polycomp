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
This package computes densities in 1D, 2D, and 3D systems for mixtures of linear
polymers and solvents, with explicit handling of charged species. 
It also computes observables such as free energy, pressure, chemical potentials, and
structure factors. 
By sampling the underlying field theory, it correctly predicts charged interactions like
coacervation that are important for biomolecular systems. 

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

`PolyComp` has been used to simulate biological systems examining mRNA encapsulation 
for drug delivery [@pertCoacervationDrivesMorphological2025], formation of nanoparticle
superlattices [@yeNanoparticleSuperlatticesAssembled2025] and as a base for the
development of machine-learned acceleration to general polymer field theories
[@pertScalingFieldTheoreticSimulation2025]. 
As the code is written entirely in Python, it provides an accessible platform for 
performant polymer simulations for any researcher familiar with the language.
The source code for PolyComp has been
archived to Zenodo with the linked DOI: [@polycomp_zenodo]

# Acknowledgements

We acknowledge contributions from the many members of the Rotskoff Group who have helped
with conceptualization, debugging, and optimization of this project, in particular 
Sherry Li, Clay Batton, Andy Mitchell, and Nicholas Juntunen. 

# References

