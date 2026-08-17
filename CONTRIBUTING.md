# Contributing to Polycomp

We welcome community contributions, bug reports, and questions. For long-term continuity and public visibility, GitHub remains the best place to contribute or seek help. 

## Reporting Bugs
If you encounter an error, installation failure, or unexpected behavior, please [open an issue](https://github.com/rotskoff-group/polycomp/issues). To help us resolve it quickly, please include:
* Your operating system and Python version.
* Your hardware specifications (e.g., specific GPU architecture and CUDA version).
* A minimal, reproducible script that triggers the error.
* The full traceback of the error.

## Seeking Support
If you have questions about how to use the software, set up a specific simulation, or understand the underlying methodology, please [open an issue](https://github.com/rotskoff-group/polycomp/issues) and apply the `Question` label. Please reference the technical papers for mathematical details of methods before reaching out. 


## Contributing Code
We gladly accept Pull Requests (PRs) for bug fixes, new features, and documentation improvements. 
1. Fork the repository and create a new branch for your feature.
2. Write clean, documented code and include tests for new functionality. Please note that we cannot do automatic testing, so please ensure that your code passes all tests (`cd test && python tests.py`) locally before submitting. 
3. Ensure your environment matches the provided `polycomp.yml`.
4. Submit a PR describing your changes. If your PR addresses an open issue, please reference it (e.g., "Addresses #12") in the description.
