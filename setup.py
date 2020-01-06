from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("nbc", sources=["nbc.pyx"], include_dirs=[numpy.get_include()],
              language="c++"),
    Extension("neighbourhood", sources=["neighbourhood.pyx"], include_dirs=[numpy.get_include()],
              language="c++")
]

setup(
    ext_modules=cythonize(extensions, annotate=True, compiler_directives={'language_level': "3"})
)
