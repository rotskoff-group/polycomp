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

`PolyComp` has been used to simulate biological systems examining mRNA encapsulation 
for drug delivery [@pertCoacervationDrivesMorphological2025], formation of nanoparticle
superlattices [@yeNanoparticleSuperlatticesAssembled2025] and as a base for the
development of machine-learned acceleration to general polymer field theories
[@pertScalingFieldTheoreticSimulation2025]. 
As the code is written entirely in Python, it provides an accessible platform for 
performant polymer simulations for any researcher familiar with the language.
The source code for PolyComp has been
archived to Zenodo with the linked DOI: [@polycomp_zenodo].
Detailed derivations of methods are available in a previous publication 
[@pertCoacervationDrivesMorphological2025].

# Acknowledgements

We acknowledge contributions from the many members of the Rotskoff Group who have helped
with conceptualization, debugging, and optimization of this project, in particular 
Sherry Li, Clay Batton, Andy Mitchell, and Nicholas Juntunen. 

# References

