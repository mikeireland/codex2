from setuptools import Extension
from setuptools import setup
import numpy as np

module = Extension("codex2rt", sources=["codex2rt.c"], extra_compile_args=["-Wall"], \
    include_dirs=[np.get_include()])

setup(
    name="codex2rt",
    version="1.0",
    description="An example Python C extension module",
    url="https://github.com/mikeireland/codex2",
    author="Mike Ireland",
    author_email="michael.ireland@anu.edu.au",
    license="GPL3",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    ext_modules=[module],
#    test_suite="tests",
)