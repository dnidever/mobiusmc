#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, find_packages

setup(name='mobiusmc',
      version='1.0.0',
      description='Sampler for periodic signals',
      author='David Nidever',
      author_email='dnidever@montana.edu',
      url='https://github.com/dnidever/mobiusmc',
      packages=find_packages(exclude=["tests"]),
      scripts=['bin/mobiusmc'],
      install_requires=['numpy','astropy(>=4.0)','scipy','dlnpyutils(>=1.0.3)','dill','emcee','corner']
)
