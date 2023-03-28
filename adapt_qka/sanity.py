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

"""Check for installed dependencies"""

###########################################################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
  import matplotlib
except ImportError:
  print(" \
      ###################################\n \
      WARNING:\n \
      >> This package depends on Matplotlib version 3.6.2.\n \
      >> To install Matplotlib, run: $ python3 -m pip install -U matplotlib==3.6.2\n \
      ###################################\n"
       )
try:
  import numpy
except ImportError:
  print(" \
      ###################################\n \
      WARNING:\n \
      >> This package depends on NumPy version 1.23.5.\n \
      >> To install NumPy, run: $ python3 -m pip install -U numpy==1.23.5\n \
      ###################################\n"
       )
try:
  import sklearn
except ImportError:
  print(" \
      ###################################\n \
      WARNING:\n \
      >> This package depends on scikit-learn version 1.2.1.\n \
      >> To install scikit-learn, run: $ python3 -m pip install -U scikit-learn==1.2.1\n \
      ###################################\n"
       )
try:
  import pandas
except ImportError:
  print(" \
      ###################################\n \
      WARNING:\n \
      >> This package depends on Pandas version 1.5.3.\n \
      >> To install Pandas, run: $ python3 -m pip install -U pandas==1.5.3\n \
      ###################################\n"
       )
try:
  import pennylane
except ImportError:
  print(" \
      ###################################\n \
      WARNING:\n \
      >> This package depends on pennylane version 0.28.0\n \
      >> To install pennylane, run: $ python3 -m pip install -U pennylane==0.28.0\n \
      ###################################\n"
       )
  raise
try:
  import qiskit_ibm_provider
except ImportError:
  print(" \
      ###################################\n \
      WARNING:\n \
      >> This package depends on qiskit_ibm_provider version 0.4.0.\n \
      >> To install qiskit_ibm_provider, run: $ python3 -m pip install -U qiskit_ibm_provider==0.4.0\n \
      ###################################\n"
       )
try:
  import torch
except ImportError:
  print(" \
      ###################################\n \
      WARNING:\n \
      >> This package depends on PyTorch version 1.13.0.\n \
      >> To install PyTorch, run: $ python3 -m pip install -U torch==1.13.0\n \
      ###################################\n"
       )
try:
  import pytest
except ImportError:
  print(" \
    ###################################\n \
    WARNING:\n \
    >> This package depends on pytest version 7.2.2\n \
    >> To install pytest, run: $ python3 -m pip install -U pytest==7.2.2\n \
    ###################################\n"
        )
try:
  import argparse
except ImportError:
  print(" \
    ###################################\n \
    WARNING:\n \
    >> This package depends on argparse version 1.4.0\n \
    >> To install argparse, run: $ python3 -m pip install -U argparse==1.4.0\n \
    ###################################\n"
        )
  raise
'''
try:
  import pylatexenc
except ImportError:
  print(" \
      ###################################\n \
      WARNING:\n \
      >> This package depends on pylatexenc version 2.10.\n \
      >> To install pylatexenc, run: $ python3 -m pip install -U pylatexenc==2.10\n \
      ###################################\n"
       )
try:
  import watermark
except ImportError:
  print(" \
      ###################################\n \
      WARNING:\n \
      >> This package depends on watermark version 2.3.1.\n \
      >> To install watermark, run: $ python3 -m pip install -U watermark==2.3.1\n \
      ###################################\n"
       )
try:
  import cuquantum
except ImportError:
  print(" \
    ###################################\n \
    WARNING:\n \
    >> This package depends on cuquantum\n \
    >> To install cuquantum, run: $ python3 -m pip install -U cuquantum\n \
    ###################################\n"
        )
try:
  import tensorflow 
except ImportError:
  print(" \
      ###################################\n \
      WARNING:\n \
      >> This package depends on TensorFlow  version 2.11.0.\n \
      >> To install TensorFlow, run: $ python3 -m pip install -U tensorflow==2.11.0\n \
      ###################################\n"
       )
  raise
'''
