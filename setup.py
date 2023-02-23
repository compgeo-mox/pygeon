#!/usr/bin/env python

import os.path
from glob import glob
from os.path import basename, splitext

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


with open("requirements.txt") as f:
    required = f.read().splitlines()


#long_description = read("Readme.rst")

setup(
    name="pygeon",
    version="0.2.0",
    license="GPL",
    keywords=["a python package for geo-numerics"],
    author="Enrico Ballini, Wietse M. Boon, Alessio Fumagalli, Anna Scotti",
    install_requires=required,
    description="A python package for geo-numerics",
    long_description="", #long_description,
    maintainer="Alessio Fumagalli",
    maintainer_email="alessio.fumagalli@polimi.it",
    platforms=["Linux", "Windows"],
    package_data={"pygeon": ["py.typed"]},
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[
        os.path.splitext(os.path.basename(path))[0] for path in glob("src/*.py")
    ],
    zip_safe=False,
)
