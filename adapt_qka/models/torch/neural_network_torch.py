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
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetworkTorch(nn.Module):
  '''Fully connected perceptron neural network with Torch's Funtional API.'''
  def __init__(self, NUM_ANGLES):
    super(NeuralNetworkTorch, self).__init__()
    self.in_=NUM_ANGLES
    self.out=self.in_
    self.linear1 = nn.Linear(self.in_, 32)
    self.linear2 = nn.Linear(32, 4)
    self.linear3 = nn.Linear(4, self.out)
    self.dropout = nn.Dropout(p=0.25)
      
  
    def forward(self, data):
      '''
      Method to define the forward pass of the neural network.
      
      Args:
        - data (numpy.ndarray): data input to be fed to the neural network.
      Returns:
        - out (numpy.ndarray): array of new optimized parameters (euler angles).
      '''
      l1 = F.relu(self.linear1(data))
      l1 = self.dropout(l1)

      l2 = F.relu(self.linear2(l1))
      l2 = self.dropout(l2)
      
      out = self.linear3(l2.view(-1, 4))
      return out
    
class CNNTorch(nn.Module):
    '''Convolutional neural network with Torch's Funtional API.'''
    def __init__(self, FEATURES):
      super(NeuralNetworkTorch, self).__init__()
      self.in_=FEATURES
      self.out=self.in_
      
      self.pool = nn.MaxPool2d((2, 2))
            
      self.conv1 = nn.Conv2d(in_channels=1,
                             out_channels=32,
                             kernel_size=(3, 3),
                             padding=1,
                             stride=1)
      
      self.bn1 = nn.BatchNorm2d(32)

      self.conv2 = nn.Conv2d(in_channels=32,
                             out_channels=16,
                             kernel_size=(3, 3),
                             padding=1,
                             stride=1)
      
      self.bn2 = nn.BatchNorm2d(16)

      self.linear1 = nn.Linear(16, 4)
      self.bn3 = nn.BatchNorm2d(4)
      
      self.linear2 = nn.Linear(4, self.out)
      
    def forward(self, data):
      '''
      Method to define the forward pass of the neural network.
      
      Args:
        - data (numpy.ndarray): data input to be fed to the neural network.
      Returns:
        - out (numpy.ndarray): array of new optimized parameters (euler angles).
      '''
      l1 = F.relu(self.conv1(data))
      l1 = self.pool(self.bn1(l1))
      
      l2 = F.relu(self.conv2(l1))
      l2 = self.pool(self.bn2(l2))
      
      conv_out = l2.view(-1, 16)
      
      l1 = F.relu(self.linear1(conv_out))
      l1 = self.bn3(l1)
      
      l2 = F.relu(self.linear2(l1))
      
      out = self.linear2(l2)
      return out
    
class RnnTorch():
  '''Recurrent Neural Network.'''
  def __init__(self, NUM_ANGLES):
    pass