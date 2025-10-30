## Testing the `polycomp` Codebase

Instructions for verifying installation of polycomp was successful and for ensuring that 
changes are safe with respect to previous work. Some tests are snapshot tests, changes 
that modify the random seeding of the code or make other changes to integration schemes
are likely to break these tests. 

---

### Hardware & Software Requirements

*   **Hardware**: An NVIDIA GPU.
*   **Software**: A compatible CUDA toolkit (e.g., version 11.8 or newer) 
and the Conda package manager.

---

### Setup Instructions

See main README file. 

---

### Running the Tests

The test suite is built using Python's standard `unittest` framework. 
Run test from `polycomp/tests/` (current directory) as 

```bash
python test.py
```

Tests not passing are most likekly caused by changes to the ranom seeding or integration
if code is being updated
(will break early tests) or could be due to small machine precision errors for some 
later tests. 
Other failures likely indicate that something is wrong with the code or your 
instalation of it.
