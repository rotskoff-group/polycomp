#Overview

This section is intended to serve as a broad look at the mathematical backing for the
code and articulate some of the design decisions we have made. The best general 
work on polymer field theories are Glenn Fredrickson's book "[The Equilibrium Theory 
of Inhomogeneous Polymers](https://academic.oup.com/book/34783)." This will be a shorter
description of the math behind the code, with a particular focus on choices we have made
in this implementation of the code. Full mathematical descriptions can be found in our 
corresponding paper *Insert paper once we have it*.

At a high level, the overall polymer field theory method operates by sampling over 
chemical potential fields and finding the corresponding real density and partition 
functions associated with any polymer being simulated. By iterating on our chemical 
potential fields, we can find free energy minima and sample around them. From there we 
can directly compute other observables such as chemical potential, pressure, structure
factor, etc. 

In general the code acts on fields $\{\mu_i\}/\{\psi_i\}/\varphi$ and densities 
$\{\rho_i\}$. Actual sampling and fictitious dynamics modify the fields to try to sample
an equilibrium distribution that can be used to evaluate observables. 

##Core assumptions and limitations

- Guassian Chain Assumption

    We model all polymers as Gaussian chains. This means there is no stiffness to the 
polymers. The model assumes that polymers can occupy any space curve. 

- Flory-Huggin and Charged Interactions

    All interactions except for charged interactions are represented as pairwise 
Flory-Huggins interactions. 

- Soft Repulsions
    
    For regularization purposes, all interactions are smeared, which can be interpreted
as either having a distributed mass density or a softer interaction than the FH 
interaction would normally imply. 

- No True Dynamics

    Dynamics in polymer field theories is a complicated issue, and while the code does 
run fictitious dynamics to equilibrate the simulation, we currently only can evaluate 
equilibrium configurations. 

