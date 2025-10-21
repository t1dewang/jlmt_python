from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

source_files = [
    'ipem_auditory_model.pyx',
    'IPEMAuditoryModel.c',
    './audiprog/Audimod.c',
    './audiprog/AudiProg.c',
    './audiprog/cpu.c',
    './audiprog/cpupitch.c',
    './audiprog/decimation.c',
    './audiprog/ecebank.c',
    './audiprog/filterbank.c',
    './audiprog/Hcmbank.c',
    './library/command.c',
    './library/filenames.c',
    './library/pario.c',
    './library/sigio.c'
]

ext = Extension(
    name="ipem_auditory_model",
    sources=source_files,
    include_dirs=[".", "./audiprog", "./library", np.get_include()],
    define_macros=[("_CRT_SECURE_NO_WARNINGS", None)], 
    extra_compile_args=["-std=c99", "-O2", "-Wall"],   
    language="c",
)

setup(
    name="ipem_auditory_model",
    ext_modules=cythonize(
        [ext],
        compiler_directives={"language_level": "3"},
        annotate=True, 
        verbose=True
    ),
)
