from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("az_mcts", ['AZMCTS.pyx'], include_dirs=[np.get_include()]),
                   Extension("board_ctrl", ['board_ctrl.pyx'], include_dirs=[np.get_include()])],
)