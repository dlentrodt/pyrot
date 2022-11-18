pyrot
=====

[D. Lentrodt](https://github.com/dlentrodt)

(Fabry-)*pyrot* is an open-source python package for the physics of one-dimensional (1D)
Fabry-Pérot cavities containing two-level atoms interacting with the light-field.

Currently, it implements the following features:
- **Transfer matrix formalism** (also known as Parratt's formalism) to calculate cavity scattering.
- **Linear dispersion theory** to compute linear scattering (i.e. in the weak driving limit) when two-level atoms are present inside the cavity.
- Calculation of the cavity **Green's function** via a recursive algorithm.
- Via the Green's function, one can set up **Markovian Master equations** for the atom ensemble at weak light-matter coupling.

Note that this software and its algorithms are mainly designed with the goal
of transparent physics and to illustrate theoretical concepts. It is *not*
designed for realistic practical applications or numerical efficiency.

<p align="center">
  <img src="https://github.com/dlentrodt/pyrot/blob/master/images/illu_readme.png" />
</p>

Installation
------------

It is recommended to first create a virtual environment.

*pyrot* is available on `pip` and can be installed using

```bash
pip install pyrot
```

Documentation
-------------

A detailed documentation is currently not available and will be added at a
later time. The current main documentation is given in form of explanatory jupyter notebooks,
which can be found in `demo/`.


As a short documentation on the underlying algorithms, mathematical details and physics background
is given in the following resources:
- **Parratt's formalism** is particularly known from x-ray scattering on thin films.
It essentially solves Maxwell's equations for a stack of layers with a given refractive index.
The formalism works recursively by adding up all the paths between the layer interfaces, whose response
is encoded in their Fresnel coefficients. [see https://doi.org/10.1103/PhysRev.95.359]
- The **transfer matrix formalism** can be considered a rewriting of Parratt's formalism, which packs the
recursively algorithm into a Matrix multiplication. In addition, it allows for polarization. However, this
package deals with normal incidence only (for now) and polarization is ignored (for now).
[see e.g. https://en.wikipedia.org/wiki/Transfer-matrix_method_(optics) and references therein]
- **Linear dispersion theory** is a method to encode the response of level schemes (such as atoms) into a
frequency dependent refractive index. This simplification is possible at *weak excitation*, where the
level schemes behave like classical oscillators, such that the *response is linear in the
excitation field*. It relies on the approximation $\langle\hat{a}(t)\hat{\sigma}^-(t)\rangle\approx-\langle\hat{a}(t)\rangle$ or similar formulations.
[see e.g. https://doi.org/10.1103/PhysRevLett.64.2499, https://doi.org/10.1103/PhysRevA.93.012120, https://doi.org/10.1103/PhysRevX.10.011008,
https://doi.org/10.1103/PhysRevResearch.2.023396]
- The classical electromagnetic **Green's tensor** is defined by the equation
$$[\nabla\times\nabla\times - \frac{\omega^2}{c^2} \varepsilon(\mathbf{r}, \omega)] \mathbf{G}(\mathbf{r}, \mathbf{r}', \omega) = \delta(\mathbf{r} - \mathbf{r}') \,,$$
Here, we consider the 1D special case, which can be regarded the normal incidence component of a layer stack in Fourier space
and is available analytically via a recursive algorithm [https://doi.org/10.1103/PhysRevA.51.2545].
- The Green's function can e.g. be used to set up Markovian Master equations for the atom ensemble in the weak coupling limit. [see https://arxiv.org/abs/0902.3586]

For a summary of these methods and their connection see also https://doi.org/10.11588/heidok.00030671.

Citing *pyrot*
--------------

The package was released together with https://arxiv.org/abs/2107.11775 and is used therein.
If you use *pyrot* in your research, please cite this preprint or the corresponding journal article once available.

*pyrot* further builds on techniques developed in https://doi.org/10.1103/PhysRevX.10.011008,
https://doi.org/10.1103/PhysRevResearch.2.023396 and summarized in https://doi.org/10.11588/heidok.00030671. Please consider
citing these papers if you find them useful.




