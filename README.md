pyrot
=====

[D. Lentrodt](https://github.com/dlentrodt)

*pyrot* is an open-source python software package for the physics of one-dimensional (1D)
Fabry-Pérot cavities containing two-level atoms interacting with the light-field.

Currently, it implements the following features:
- **Transfer matrix formalism** (also known as Parratt's formalism) to calculate cavity scattering.
- **Linear dispersion theory** to compute linear scattering (i.e. in the weak driving limit) when two-level atoms are present inside the cavity.
- Calculation of the cavity **Green's function** via a recursive algorithm (https://doi.org/10.1103/PhysRevA.51.2545). The Green's function can be used to set up Markovian Master equations for the atom ensemble in the weak coupling limit.


Note that this software and its algorithms are mainly designed with the goal
of transparent physics and to illustrate theoretical concepts. It is *not*
designed for realistic practical applications or numerical efficiency.

Installation
------------

It is recommended to first create a virtual environment.

*pyrot* can then be installed using

    pip install pyrot-0.0.0.tar.gz

where the tar-ball can be found in the distributions directory `dist/`.

Documentation
-------------

A detailed documentation is currently not available and will be added at a
later time. Explanatory simple examples can be found in `demo/`.
For mathematical details and physics background please refer to
https://arxiv.org/abs/2107.11775 and https://doi.org/10.1103/PhysRevX.10.011008.

