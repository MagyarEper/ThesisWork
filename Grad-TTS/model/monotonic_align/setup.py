from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os

# Get the directory where this setup.py file is located
this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
  name = 'monotonic_align',
  ext_modules = cythonize(
      os.path.join(this_dir, "core.pyx"),
      language_level=3
  ),
  include_dirs=[numpy.get_include()]
)