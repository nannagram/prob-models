from setuptools import setup, find_packages
import os

__version__ = '0.0.1'
NAME = 'prob_models'
AUTHOR = 'BrainHack'
MAINTEINER = 'BrainHackers'

setup(
    name=NAME,
    version=__version__,
    packages=find_packages(),
    author=AUTHOR,
    maintainer=MAINTEINER,
    description="A package for probabilistic models",
    license='None',
    install_requires=['numpy',
                      'scipy',
                      'matplotlib',
                      'scikit-learn']
)
