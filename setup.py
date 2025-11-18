"""
setup.py — Build script for JLMT Python Package
===============================================

This script builds the JLMT C extension (`ipem_auditory_model.c`) and installs
the Python package located under the `src/` directory.

HOW TO USE (To build the C extension)
-------------------------------------

1. Open a terminal in the project root directory:

       C:/Users/.../jlmt_python>

2. Run the build command:

       python setup.py build_ext --inplace

   This will:
       - Compile the C extension into a `.pyd` (Windows)
       - Generate a file named like:
             ipem_auditory_model.cp39-win_amd64.pyd

3. IMPORTANT: Move the generated binary into the Python package directory:

       src/jlmt_py/

   Because inside `calc_ani.py` you import it with:

       import ipem_auditory_model

   Python *only* searches inside `jlmt_py` for this module.

4. (Optional) Install the whole package:

       pip install .

NOTES
-----
- The generated `.pyd` is *not portable* (works only on the same OS + Python version).
- This setup script is intended for research use and local builds.
- For collaborators, they simply run `setup.py build_ext --inplace` on their machine.

"""

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
import os

source_files = [
    os.path.join("src", "c_code", "ipem_auditory_model.pyx"),
    os.path.join("src", "c_code", "IPEMAuditoryModel.c"),
    os.path.join("src", "c_code", "audiprog", "Audimod.c"),
    os.path.join("src", "c_code", "audiprog", "AudiProg.c"),
    os.path.join("src", "c_code", "audiprog", "cpu.c"),
    os.path.join("src", "c_code", "audiprog", "cpupitch.c"),
    os.path.join("src", "c_code", "audiprog", "decimation.c"),
    os.path.join("src", "c_code", "audiprog", "ecebank.c"),
    os.path.join("src", "c_code", "audiprog", "filterbank.c"),
    os.path.join("src", "c_code", "audiprog", "Hcmbank.c"),
    os.path.join("src", "c_code", "library", "command.c"),
    os.path.join("src", "c_code", "library", "filenames.c"),
    os.path.join("src", "c_code", "library", "pario.c"),
    os.path.join("src", "c_code", "library", "sigio.c"),
]


ext = Extension(
    name="jlmt_py.ipem_auditory_model",
    sources=source_files,
    include_dirs=[
        os.path.join("src", "c_code"),
        os.path.join("src", "c_code", "audiprog"),
        os.path.join("src", "c_code", "library"),
        np.get_include()
    ],
    define_macros=[("_CRT_SECURE_NO_WARNINGS", None)],
    extra_compile_args=["-O2"], 
    language="c",
)

setup(
    name="jlmt_python",
    version="0.1.0",
    author="Tao Wang",
    description="JLMT auditory model and tonal space computation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(
        [ext],
        compiler_directives={"language_level": "3"},
        annotate=False,
        verbose=False
    ),
    include_package_data=True,
)