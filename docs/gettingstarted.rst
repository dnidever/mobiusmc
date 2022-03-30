***************
Getting Started
***************



Introduction
============

This is the Gaussian Decomposition software described in `Nidever et al. (2008) <https://ui.adsabs.harvard.edu/abs/2008ApJ...679..432N/abstract>`_ following the algorithm from `Haud (2000) <https://ui.adsabs.harvard.edu/abs/2000A%26A...364...83H>`_.  While it was designed to be used for HI spectra, it can be used for other types of data like Halpha spectra.


Note that I wrote this software as a first-year graduate student back in 2005 and I haven't had much time to optimize it since then.  One of my goals is to speed it up.




Overview of Python code: :ref:`Python Overview`

Overview of IDL code: :ref:`IDL Overview`



.. _Python Overview:

Python Overview
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
	
.. |gaussfitfig| image:: gaussfit.png
  :width: 800
  :alt: Gaussian Fit to Spectrum

|gaussfitfig|


	
To fit an entire datacube, you can either give the driver code a datacube object you have already created or give it a FITS filename.

.. code-block:: python

	from gaussdecomp import cube,driver
	# Load the cube first
	datacube = cube.Cube.read('mycube.fits')
	gstruc = driver.driver(datacube)

	# Give it the FITS filename
	gstruc = driver.driver('mycube.fits')


See the :doc:`examples` page for some examples of how Python |gaussdecomp| runs.
	

.. _IDL Overview:

IDL Overview
============


The software was originally run on the Leiden-Argentine-Bonn all-sky HI survey and there are hard-coded settings that need modification for certain datasets.  My plan is to modify the code at some point in the future to be more general and to allow these values to be more configurable.  Currently, I've been making dataset-specific copies of five of the programs and modifying the settings. The programs are:

 - `gdriver.pro <https://github.com/dnidever/gaussdecomp/blob/master/pro/gdriver.pro>`_: Main driver program.  There are a few hard-coded defaults that you might want to change.
 - `gloadspec.pro <https://github.com/dnidever/gaussdecomp/blob/master/pro/gloadspec.pro>`_: Loads the cube.  You might want to make some modifications to how it loads the data.
 - `gincrement.pro <https://github.com/dnidever/gaussdecomp/blob/master/pro/gincrement.pro>`_: Increments the current position.  Depending on the data the step-size will change.
 - `parcheck.pro <https://github.com/dnidever/gaussdecomp/blob/master/pro/parcheck.pro>`_: Checks if Gaussians have "bad" parameters.  The thresholds for "bad" might need to be tweaked.
 - `setlimits.pro <https://github.com/dnidever/gaussdecomp/blob/master/pro/setlimits.pro>`_: Set limits on all of the Gaussian parameters (height, velocity, width). 
 - `hinoise.pro <https://github.com/dnidever/gaussdecomp/blob/master/pro/hinoise.pro>`_: The program that calculates the noise in each spectrum.  You should set which velocity range to use.


However, for most cubes the defaults should be okay.

   
See the :doc:`examples` page for some examples of how IDL |gaussdecomp| runs.

