***************
Getting Started
***************



Introduction
============

|mobiusmc| is a sampler for periodic signals such as variable star light curves.  It was inspired by `The Joker <https://github.com/adrn/thejoker>`_ which is a custom-built sampler for Keplerian orbits in radial velocity data.

The general idea is that both the period and phase shift (the non-linear terms) are highly sampled in a direct Monte Carlo fashion.  The linear terms are marginalized over (i.e. the best solution is found in a linear least squares sense) for each period and phase shift.  

There are three sampler classes:
 - :class:`~mobiusmc.sampler.Sampler` is a generic sampler for any periodic problem.
 - :class:`~mobiusmc.sampler.LinearModelSampler` is a sampler for a periodic problem where the model can be written as a simple linear model of the form `y = A*model() + B`.
 - :class:`~mobiusmc.sampler.VariableSampler` is a sampler for variable star lightcurves.


Generic Sampler
===============

There are five main modules:

 - :mod:`~gaussdecomp.driver`:  Decomposes all of the spectra in a datacube.
 - :mod:`~gaussdecomp.fitter`:  Does the actual Gaussian Decomposition.
 - :mod:`~gaussdecomp.cube`:  Contains the :class:`~gaussdecomp.cube.Cube` class for a data cube.
 - :mod:`~gaussdecomp.spectrum`:  Contains the :class:`~gaussdecomp.spectrum.Spectrum` class for a single spectrum.
 - :mod:`~gaussdecomp.utils`:  Various utility functions.

There is a class for data cubes called :class:`~gaussdecomp.cube.Cube` and a class for spectra called :class:`~gaussdecomp.spectrum.Spectrum`.

To fit a single spectrum you first need to create the Spectrum object.

.. code-block:: python

	from gaussdecomp import spectrum,fitter
	sp = spectrum.Spectrum(flux,vel)   # flux and velocity arrays
	out = fitter.gaussfit(sp)          # do the fitting

You can make a nice plot using :func:`~gaussdecomp.utils.gplot`.

.. code-block:: python

	from gaussdecomp import utils
	utils.gplot(vel,flux,par)


Linear Model Sampler
====================
	
.. |gaussfitfig| image:: gaussfit.png
  :width: 800
  :alt: Gaussian Fit to Spectrum

|gaussfitfig|

Variable Star Sampler
=====================

	
To fit an entire datacube, you can either give the driver code a datacube object you have already created or give it a FITS filename.

.. code-block:: python

	from gaussdecomp import cube,driver
	# Load the cube first
	datacube = cube.Cube.read('mycube.fits')
	gstruc = driver.driver(datacube)

	# Give it the FITS filename
	gstruc = driver.driver('mycube.fits')

   
See the :doc:`examples` page for some examples of how to run |mobiusmc|.

