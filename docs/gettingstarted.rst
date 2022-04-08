***************
Getting Started
***************



Introduction
============

|mobiusmc| is a sampler for periodic signals such as variable star light curves.  It was inspired by `The Joker <https://github.com/adrn/thejoker>`_ which is a custom-built sampler for Keplerian orbits in radial velocity data.

The general idea is that both the period and phase shift (the non-linear terms) are highly sampled in a direct Monte Carlo fashion.  The linear terms are marginalized over (i.e. the best solution is found in a linear least squares sense) for each period and phase shift.  The probability at each point is then used to perform rejection sampling, i.e. a random number is drawn between 0 and 1 and if the number is **less** than the probability then the point is kept otherwise it is rejected.  The period and phase offset is sampled until the minimum number of final samples has been reached.

There are three sampler classes:
 - :class:`~mobiusmc.sampler.Sampler` is a generic sampler for any periodic problem.
 - :class:`~mobiusmc.sampler.LinearModelSampler` is a sampler for a periodic problem where the model can be written as a simple linear model of the form `y = A*model(phase) + B`.
 - :class:`~mobiusmc.sampler.VariableSampler` is a sampler for variable star lightcurves.

Below, each of the three sampler classes is described in more detail.

Generic Sampler
===============

The generic :class:`~mobiusmc.sampler.Sampler` class is designed for a problem where the data is periodic in the independent variable.  The user needs to provide a function that returns the log probability value for a given set of (period,phase offset) pairs.  The sampler will highly sample the period and phase offset dimensions and perform rejection sampling until the minimum number of samples has been reached.

.. note::
   The user-defined log probability function should be able to handle arrays of period and phase offset (normally a tuple of two arrays).

The usual way to write the log probability function is using two separate log prior and log likelihood functions whose outputs are then combined to create the log probability.  Here's an example for a simple linear problem from `emcee's` `documentation <https://emcee.readthedocs.io/en/stable/tutorials/line/>`_:

.. code-block:: python

	# The log likelihood function
	def log_likelihood(theta, x, y, yerr):
	    m, b = theta
	    model = m * x + b
	    sigma2 = yerr ** 2 + model ** 2 * np.exp(2)
	    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

	# The log prior function
	def log_prior(theta):
	    m, b = theta
	    if -5.0 < m < 0.5 and 0.0 < b < 10.0:
  	        return 0.0
	    return -np.inf

	# The log probability function combines the outputs of the log prior and log likelihood functions
	def log_probability(theta, x, y, yerr):
	    lp = log_prior(theta)
	    if not np.isfinite(lp):
		return -np.inf
	    return lp + log_likelihood(theta, x, y, yerr)

This is how the sampler calls the user-defined log probability function. 

.. code-block:: python
		
	# Calculate the ln probabilty
        lnprob = log_probability([period,offset],*args,**kwargs)

Here `period` and `offset` (phase offset) are arrays, while `args` and `kwargs` are positional and keyword arguments that the user can pass to the log probability function. `args` should be a tuple that contains **at least** (x,y,yerr), but can have additional arguments.


Here's an example of how to run the :class:`~mobiusmc.sampler.Sampler` sampler.

.. code-block:: python

	from mobiusmc.sampler import Sampler
	# Instantiate sampler with the data and model function.
	samp = sampler.Sampler((x,y,yerr),modelfunc)
	# Run the sampler
	samp.run(verbose=True)
	# Make some plots
	samp.plots()

	

Linear Model Sampler
====================

The :class:`~mobiusmc.sampler.LinearModelSampler` class is designed for a probem where the data can be described by a model of the form `y = A*model(phase) + B`, where `A` is the amplitude parameter and `B` is a constant offset.  `model()` is a user-defined function that depends only on `phase`.  It must handle `phase` being an array.

The sampler highly samples period and phase offset and for each pair of values uses linear least squares to find the best amplitude and constant offset (the linear parameters):

:class:`~mobiusmc.sampler.LinearModelSampler` must be given the data (tuple of x,y,yerr) and the model function.

Here's an example of how to run the :class:`~mobiusmc.sampler.LinearModelSampler` sampler.

.. code-block:: python

	from mobiusmc.sampler import LinearModelSampler
	# Instantiate sampler with the data and model function.
	data = (x,y,yerr)
	lms = sampler.LinearModelSampler(data,modelfunc)
	# Run the sampler
	lms.run(verbose=True)
	# Make some plots
	lms.plots()

	
.. |gaussfitfig| image:: gaussfit.png
  :width: 800
  :alt: Gaussian Fit to Spectrum

|gaussfitfig|

Variable Star Sampler
=====================

The :class:`~mobiusmc.sampler.VariableSampler` class is designed to sample a variable star lightcurve (often in multiple bands) with a single template (same for all bands).  The software fits only a single amplitude of the template, but in reality the amplitude varies from band to band (larger in bluer bands, smaller in redder bands).  Therefore, it is good to input the amplitude ratios dictionary.  The software also determined the mean magnitude in each band.

:class:`~mobiusmc.sampler.VariableSampler` just be given an input catalog of information that contains the columns 'mag','err','jd','band'.  It must also be given the template which must have the columns 'mag','phase'.  Finally, `ampratios` should be a dictionary where the keys are the names of the unique bands.

Here's an example of how to run it.

.. code-block:: python

	from mobiusmc.sampler import VariableSampler
	# Instantiate the variable star sampler with the data, template and amplitude ratios (optional)
	vs = sampler.VariableSampler(data,template,ampratios=ampratios)
	# Run the sampler
	vs.run(verbose=True)
	# Make some plots
	vs.plots()

   
See the :doc:`examples` page for some examples of how to run |mobiusmc|.

