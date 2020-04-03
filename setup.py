from setuptools import setup, find_packages
import runpy
__version__ = runpy.run_path('neurofire/version.py')['__version__']


setup(name='neurofire',
      version=__version__,
      description='Toolkit for deep learning with connectomics datasets.',
      packages=find_packages(exclude=['test']))
