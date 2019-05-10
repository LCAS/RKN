# Always prefer setuptools over distutils
from setuptools import setup, Extension

# To use a consistent encoding
from codecs import open
from os import path
import numpy

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# path to forward models
fmPath = "forwardModels/"

# Install Dual Link Forward Model
_DoubleLinkForwardModel = Extension("_DoubleLinkForwardModel",
                                    [fmPath + "DoubleLinkForwardModel.i",
                                     fmPath + "DoubleLinkForwardModel.c"],
                                    include_dirs=[numpy_include])

# Install Quad Link Forward Model
_QuadLinkForwardModel = Extension("_QuadLinkForwardModel",
                                  [fmPath + "QuadLinkForwardModel.i",
                                   fmPath + "QuadLinkForwardModel.c"],
                                  include_dirs=[numpy_include],
                                  #extra_compile_args=['-fopenmp'],
                                  #extra_link_args=['-lgomp']
                                  )

setup(
    name='N Link Simulator',
    version='0.0.1',
    author='Philipp Becker',
    author_email='philippbecker93@googlemail.com',
    description='Simulators for double and quad link dynamics, taken from pypost toolbox',
    long_description=long_description,
    url='',
    license='unknown',

    classifiers=[
        'Programming Language :: Python :: 3.5',
    ],

    ext_modules=[_DoubleLinkForwardModel,
                 _QuadLinkForwardModel],
)
