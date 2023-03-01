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

"""Under development..."""

###########################################################################
import os
import argparse
import torch
import torch.optim as optim

class Train:
  '''Training loop class.'''
  def __init__(self, data):
    self.data = data
    
  def save_model(self, model):
    '''
    Saves the neural network model after training.
    '''
    save_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'model', 'torch', 'weights', 'model.weights'))
    try:
      torch.save(model.state_dict(), save_dir/model.weights)
    except RuntimeError:
      print('Directory does not exist. Creating new directory.')
      if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f'Directory {save_dir} successfully created!')
      else:
        print(f'Directory {save_dir} already exists!')
      torch.save(model.state_dict(), save_dir/model.weights)
      print('Model successfully saved.\n')
    
  def neural_network_torch(self, neural_network, epochs, cost):
    '''
    Method for the training loop of the classical neural network. The output of the neural network should be the parameter initialization for the quantum circuit.
    
    Returns:
      - angles(np.ndarray): array of optimized angles.
    '''   
    # Instantiate model with CPU or GPU if available:
    try:
      model = neural_network.cuda()
      print('\nUsing cuda GPU.\n')
    except:
      print('\nTorch not compiled with CUDA enabled.')
      model = neural_network
      print('Using CPU instead.\n')
    finally:
      print('Training the neural network...\n')
    # SGD optimzier:
    optimizer = optim.SGD(model.parameters(), lr=0.00025, momentum=0.9)
    for _ in range(epochs):
      print(f'Running training loop for epoch {_+1}/{epochs}...\n')
      loss = 0.0
      model.train() 
      print('To be continued...')
      break
    #print('Saving trained model...')  
    initialization = None      
    return initialization