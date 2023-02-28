# -*- coding: utf-8 -*-

#@title Copyright 2023.
# This code is part of adapt-qka.
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

"""QuCAI-Lab qhack2023-openhack"""

###########################################################################
# Classical Neural Network:
import keras 
import tensorflow as tf
from keras import backend as k
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint 
from keras.models import Sequential, load_model
#from keras.utils.generic_utils import get_custom_objects # Adding a new activation function to Keras
from keras.initializers import RandomNormal, LecunNormal # LecunNormal is used whenever 'selu' activation function is used.
from keras.layers import Dense, Dropout, AlphaDropout, BatchNormalization, Activation, LeakyReLU #, ELU, ReLU, # AlphaDropout is used whenever dropout regularization 
                                                                                                               # and 'selu' activation function are used.
                                                                                                
class NeuralNetworkTF():
  '''Recurrent Neural Network.'''
  def __init__(self):
    pass
  
class RnnTf():
  '''Recurrent Neural Network.'''
  def __init__(self):
    pass