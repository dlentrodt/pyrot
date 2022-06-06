# pyrot

*pyrot* is a python software package for the physics of one-dimensional (1D)
Fabry-Pérot cavities.

Note that this software package is mainly designed with the goal of transparency
and to illustrate theoretical concepts. It is not designed for numerical
efficiency or realistic practical applications.

## Installation

It is recommended to first create a virtual environment.

*pyrot* can then be installed using

    pip install pyrot-0.0.0.tar.gz

where the tar-ball can be found in the distributions directory `dist/`.

## Documentation

A detailed documentation is currently not available and will be added at a later time.

## Warning

We note that this package is **at a very preliminary stage**. While it is functional,
we may implement major structural changes in the future.

## Simple example

### Rocking curve and spectra

Let us look at a simple example: an X-ray cavity in grazing incidence.
In particular, we will use the so-called EIT configuration from https://doi.org/10.1103/PhysRevA.91.063803.

We first create the cavity system. Note that the `GrazingIncidence` class is
initialized using the *pygreenfn* version, which inherits from the `pynuss`
version. It contains all *pynuss* functions and additional ones.

```python
	import numpy as np
	import matplotlib.pyplot as plt
	import matplotlib
	import pynuss
	import pygreenfn.green_functions as pynuss_gf

	eFe  = pynuss.ResonantElement.fromTemplate('Fe57')
	mFe  = pynuss.Material.fromElement(eFe)
	mPt  = pynuss.Material.fromSymbol('Pt')
	mC   = pynuss.Material.fromSymbol('C')
	mSi  = pynuss.Material.fromSymbol('Si')

	nm = 1e-9
	Layers = [   pynuss.Layer(mPt,     3.   * nm),
	             pynuss.Layer(mC,     10.5  * nm),
	             pynuss.Layer(mFe,     3.0  * nm),
	             pynuss.Layer(mC,      6.5  * nm),
	             pynuss.Layer(mFe,     3.0  * nm),
	             pynuss.Layer(mC,     21.0  * nm),
	             pynuss.Layer(mPt,    10.0  * nm),
	             pynuss.Layer(mSi, -1)
	          ]

	# Initialize system
	Beam     = pynuss.Beam()
	Detector = pynuss.Detector(Beam)
	gr       = pynuss_gf.GrazingIncidence(Beam, Detector, Layers)
```

Let us now compute a rocking curve and a nuclear spectrum at the third rocking
minimum (where the EIT phenomenon is observed).

```python
	Theta  = np.linspace(2.25, 4.7, 202)  # mrad  -> for rocking curve
	Detuning = np.linspace(-50, 50, 201)  # γ     -> for nuclear spectrum

	# Rocking curve
	rocking_pynuss = gr.ReflectionIntensity(Theta)
	rocking_green  = gr.ReflectionIntensityFromGreen(Theta)

	# find third minimum
	Theta2 = gr.ReflectionMinimum(3)

	# nuclear spectrum in third minimum (EIT spectrum)
	subEnsembleNumber      = 3
	spectrum_nuc_pynuss    = gr.ReflectionIntensity(Theta2, Detuning)
	spectrum_nuc_green     = gr.ReflectionIntensityFromGreen(Theta2, Detuning, subensembles=None)
	spectrum_nuc_green_sub = gr.ReflectionIntensityFromGreen(Theta2, Detuning, subensembles=subEnsembleNumber)
```

In each case, we call a *pynuss* function and a corresponding `...FromGreen`
function from the *pygreenfn* package. For the rocking curve, the two are
trivially identical and the second function mainly exists as a cross-check.
For the nuclear spectrum, the Green function method introduces an ensemble
picture, which includes the thin layer approximation (see https://doi.org/10.1103/PhysRevResearch.2.023396 for details).
If the spectra do not agree sufficiently, additional subensembles can be included
via the `subensembles` parameter in `pygreenfn.ReflectionIntensityFromGreen`.

The spectra are compared in the following plot.

<details>
  <summary>Click to expand plot code...</summary>

```python
	# Plot
	plt.figure(figsize=(10,5))
	ax = plt.subplot(121, xlabel='theta [mrad]', ylabel='Reflection')
	plt.axvline(Theta2, color='k', dashes=[1,1])
	ax.plot(Theta, rocking_pynuss, label='pynuss')
	ax.plot(Theta, rocking_green, '--', label='pygreenfn')
	plt.ylim([0,1])
	plt.legend()
	ax = plt.subplot(122, xlabel=r'Detuning [$\gamma$]', ylabel=r'Spectrum')
	plt.plot(Detuning, spectrum_nuc_pynuss, label=r'pynuss')
	plt.plot(Detuning, spectrum_nuc_green, 'C3-', label=r'pygreenfn (no subensembles)')
	plt.plot(Detuning, spectrum_nuc_green_sub, 'C1--', label=r'pygreenfn ({} sub-ensembles)'.format(subEnsembleNumber))
	plt.ylim([0,1])
	plt.legend()
	plt.tight_layout()
	plt.show()
```
</details>


![Link to output figure](README_fig.png)

### Effective level scheme parameters
In addition, we can directly output the effective level scheme parameters that
are used to calculate the spectrum in the Green function method.
Note that unlike in phenomenological models, no fit is needed for this calculation!
For example, scanning over incidence angle (note that this code takes a few seconds to run).

```python
	ensN, ResIso = pynuss_gf.find_res_layer_idx(gr)
	comp_lev_shift = np.einsum('t,i,j->tij', np.zeros_like(Theta, dtype=np.complex128), ensN, ensN)
	drive_vec = np.einsum('t,i->ti', np.zeros_like(Theta, dtype=np.complex128), ensN)

	# scan
	for i_, th_ in enumerate(Theta):
	    comp_lev_shift_i_, drive_vec_i_ = gr.EffectiveLevelScheme(th_, subensembles=None)
	    comp_lev_shift[i_,:,:] = comp_lev_shift_i_
	    drive_vec[i_,:] = drive_vec_i_
	    
	# unpack level shift matrix
	Δls_1 = np.real(comp_lev_shift[:,0,0])
	Γrad_1 = -2.*np.imag(comp_lev_shift[:,0,0])
	Δls_2 = np.real(comp_lev_shift[:,1,1])
	Γrad_2 = -2.*np.imag(comp_lev_shift[:,1,1])

	Δls_12 = np.real(comp_lev_shift[:,0,1])
	Γrad_12 = -2.*np.imag(comp_lev_shift[:,0,1])
	Δls_21 = np.real(comp_lev_shift[:,1,0])
	Γrad_21 = -2.*np.imag(comp_lev_shift[:,1,0])
```


We can then plot the resulting parameters.

<details>
  <summary>Click to expand plot code...</summary>

```python
	fig = plt.figure(figsize=(10,10))
	fig.add_subplot(411, xlabel=r'$\theta$ [mrad]')
	plt.axvline(Theta2, dashes=[4,2], color='k', label='EIT incidence angle (third minimum)')
	plt.plot(Theta, rocking_pynuss, 'k-', label='rocking curve')
	plt.autoscale(enable=True, axis='x', tight=True)
	plt.ylim([0,1])
	plt.legend(loc=1)
	fig.add_subplot(412, xlabel=r'$\theta$ [mrad]')
	plt.axhline(0, color='k', lw=1)
	plt.axvline(Theta2, dashes=[4,2], color='k')
	plt.plot(Theta, Δls_1, label='Collective Lamb shift of layer 1 ($\delta_{1}$)')
	plt.plot(Theta, Γrad_1, label='Superradiance of layer 1 ($\gamma_{1}$)')
	plt.autoscale(enable=True, axis='x', tight=True)
	plt.legend()
	fig.add_subplot(413, xlabel=r'$\theta$ [mrad]')
	plt.axvline(Theta2, dashes=[4,2], color='k')
	plt.axhline(0, dashes=[2,2], color='k')
	plt.plot(Theta, Δls_2, label='Collective Lamb shift of layer 2 ($\delta_{2}$)')
	plt.plot(Theta, Γrad_2, label='Superradiance of layer 2 ($\gamma_{2}$)')
	plt.autoscale(enable=True, axis='x', tight=True)
	plt.legend()
	fig.add_subplot(414, xlabel=r'$\theta$ [mrad]')
	plt.axvline(Theta2, dashes=[4,2], color='k')
	plt.axhline(0, dashes=[2,2], color='k')
	plt.plot(Theta, Δls_12, label=r'Coherent layer coupling ($\delta_{12}$)')
	plt.plot(Theta, Γrad_12, label=r'Incoherent layer coupling ($\gamma_{12}$)')
	plt.autoscale(enable=True, axis='x', tight=True)
	plt.legend()
	plt.tight_layout()
	plt.show()
```
</details>

![Link to output figure](README_fig2.png)

For more scientific details see https://doi.org/10.1103/PhysRevResearch.2.023396.

## Further examples

Further example scripts can be found in the `demo/` directory.
