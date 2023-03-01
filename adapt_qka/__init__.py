# -*- coding: utf-8 -*-

#@title Copyright 2023.
# This code is part of adapt_qka.
#
# (C) Copyright QuCAI-Lab, 2023.
#
# This code is licensed under the Creative Commons Zero v1.0 Universal License. 
# You may obtain a copy of the LICENSE.md file in the root directory
# of this source tree or at https://creativecommons.org/publicdomain/zero/1.0/.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""QuCAI-Lab adapt_qka"""

###########################################################################

# Sanity Check 
from . import sanity 

# Core Dependencies 
import argparse, os, sys, numpy as np, matplotlib as plt, sklearn as sk, tensorflow as tf, torch, pylatexenc, watermark, pandas as pd, pennylane as qml

###########################################################################
VERSION_PATH = os.path.join(os.path.dirname(__file__), "VERSION.txt")
with open(VERSION_PATH, "r") as version_file:
  VERSION = version_file.read().strip()
  
__name__ = "adapt_qka"
__version__ = VERSION
__status__ = "Development"
__homepage__ = "https://github.com/QuCAI-Lab/adapt-qka"
__authors__ = "Lucas Camponogara Viera and 陳宏宇."
__license__ = "Apache 2.0"
__copyright__ = "Copyright QuCAI-Lab 2023"

# About
def about():
  """Function to display the adapt_qka project information."""
  print(" \
    #############################################################################################\n \
                                            QHack2023 Open Hackathon:\n \
    >>                      Adaptive Quantum Kernel Alignment for data Classification \n \
    #############################################################################################\n"
     )
  print(f"{__copyright__}")
  print(f"Name: {__name__}")
  print(f"Version: {__version__}")
  print(f"Status: {__status__}")
  print(f"Home-page: {__homepage__}")
  print(f"Authors: {__authors__}")
  print(f"License: {__license__}")
  print(f'Requires: python=={sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}, numpy=={np.__version__}, matplotlib=={plt.__version__}, pylatexenc=={pylatexenc.__version__}, argparse=={argparse.__version__}, watermark=={watermark.__version__}, pandas=={pd.__version__}, scikit-learn=={sk.__version__}, pytorch=={torch.__version__}, tensorflow=={tf.__version__}, cudatoolkit=={torch.version.cuda}, pennylane=={qml.__version__}')

###########################################################################

# Simulation
from ._main.train_nn import Train
from ._main.qka import AdaptQKA
from ._main.preprocessing import preprocessing

from .models.tensorflow.neural_network_tensorflow import NeuralNetworkTF, RnnTf

from .models.torch.neural_network_torch import NeuralNetworkTorch, CNNTorch, RnnTorch