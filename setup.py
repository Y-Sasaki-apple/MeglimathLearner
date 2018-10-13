from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ 
                    Extension("board", ['board_ctrl.pyx'], include_dirs=[np.get_include()]),
                    Extension("alphazero_net", ['AZNet.pyx'], include_dirs=[np.get_include()]),
                    Extension("game", ['game_ctrl.pyx'], include_dirs=[np.get_include()]),
                    Extension("pl", ['player.pyx'], include_dirs=[np.get_include()]),
                    Extension("ut", ['util.pyx'], include_dirs=[np.get_include()])
                   ],
)