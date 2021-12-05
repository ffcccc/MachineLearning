from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import eigency

extensions = [
    Extension("cpp.distance", ["cpp/distance.pyx"],
        include_dirs = [".", "cpp"] + eigency.get_includes(),  # include_dirs = [".", "module-dir-name", 'path-to-own-eigen'] + eigency.get_includes(include_eigen=False)
        extra_compile_args=['/O2', '/favor:blend', "/DEBUG:NONE", "-DNDEBUG"]
    ),
]

dist = setup(
    name = "cpp",
    version = "1.0",
    ext_modules = cythonize(extensions),
    packages = find_packages()
)