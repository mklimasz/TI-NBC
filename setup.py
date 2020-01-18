from setuptools import dist, Extension, find_packages, setup

dist.Distribution().fetch_build_eggs(["Cython==0.29.14", "numpy==1.17.3"])

from Cython.Build import cythonize
import numpy

requirements = [
    "absl-py==0.8.1",
    "Cython==0.29.14",
    "numpy==1.17.3",
    "pandas==0.25.3",
    "sortedcontainers==2.1.0",
]

extensions = [
    Extension("nbc.clustering", sources=["nbc/clustering.pyx"], include_dirs=[numpy.get_include()]),
    Extension("nbc.neighbourhood", sources=["nbc/neighbourhood.pyx"], include_dirs=[numpy.get_include()]),
]

setup(
    name="nbc",
    url="https://github.com/mklimasz/TI-NBC",
    version="0.0.1",
    install_requires=requirements,
    author_email="mk.klimaszewski@gmail.com",
    packages=find_packages(exclude=["tests"]),
    entry_points={"console_scripts": ["nbc = nbc.main:main"]},
    ext_modules=cythonize(extensions, annotate=True, compiler_directives={'language_level': "3"}),
)
