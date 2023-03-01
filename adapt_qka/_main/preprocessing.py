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

"""QuCAI-Lab qhack2023-openhack"""

###########################################################################
# Dataset:
import os
import random
import numpy as np
import pandas as pd
#import json
#import pickle 
#import codecs
#from pathlib import Path

# Function annotation:
from typing import Tuple, Dict#, Callable, Optional, Union, List

def clean(file) -> Tuple[np.ndarray, np.ndarray]: # https://docs.python.org/3/library/typing.html#typing.Tuple
  '''
  Converting features and labels from a .txt file to numpy.ndarray with Pandas.
  
  Args:
    - file (.txt): text file containing the dataset with features and labels.
    
  Returns:
    - output (tuple): tuple containing features and labels of type 'numpy.ndarray'.
  '''   
  dataset_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'dataset', file))
  # Reading dataset in pandas dataframe. The parameter dtype='unicode' is used when the values of the dataframe have different datatypes.
  df = pd.read_csv(dataset_dir, header=None, dtype='unicode')
  # [df.iat[i, 4] for i in range(0, df.shape[0]) if df.iat[i, 4] != df.iat[i-1, 4]] == df[4].unique().tolist() >>> True
  
  x=df[4].unique().tolist() # ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
  for word in x:
    df.replace(word, int(x.index(word)), inplace=True)
  df=df.astype(np.float32)
  
  features = df.iloc[:, :4].to_numpy().astype(np.float32)
  labels = df.iloc[:, 4:5].to_numpy().astype(np.int8).reshape(150)
  
  #print(f'\n Input Data:\n{df.head()}\n')
  print(f'\nRunning clean() function on {file} file:\n>>> features.shape = {features.shape}.\n>>> labels.shape = {labels.shape}\n')
  output = features, labels
  
  return output


def one_hot(target: np.ndarray) -> np.ndarray:
  '''
  One-hot encoding of labels for the Neural Network only, not for scikit-learn SVM.

  Args:
      - target (numpy.ndarray): the array of labels. 

  Returns:
      - outputs (numpy.ndarray): the array of labels in one-hot encoding format.

  Example:
    >>> one_hot(np.array([0,1,3]))
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 0., 1.]])
  '''
  print('\nRunning one_hot() function on labels:')
  no_classes = np.max(target)+1 # Total number of classes.
  new = np.zeros((target.shape[0], no_classes)) 
  for i, x in enumerate(target): 
    new[i,x] = 1 
  return new


def min_max_norm(x: np.ndarray) -> np.ndarray:
  '''
  This function computes the Min-Max feature scaling (normalization) for either a vector or a matrix.

  Args:
      - x (numpy.ndarray): the n-dimensional tensor (array) containing 'no. of rows' samples where each sample has 'no. of columns' features. 

  Returns:
      - norm_X (numpy.ndarray): the normalized dataset containing 'no. of rows' samples where each sample has 'no. of columns' features.

  Example:
    >>> min_max_norm(np.arange(0,2,.5).reshape(2,2))
    array([[0., 0.],
           [1., 1.]])
    >>> min_max_norm(np.array([[1,2,3]]))
    array([[0., 0.5, 1.]])
  '''
  print('Running min_max_norm() function on features:')
  norm_X = np.zeros(x.shape) # Placeholder.
  for i, j in [x.shape]:     # Unpack the values from the tuple of the 2-D numpy array.
    if i==1 or j==1:         # Check wheter one of the values is 1. Check if the input data is a vector or a matrix.
      return (x-np.min(x))/(np.max(x)-np.min(x)) # Normalization. Replace each element of the array by its normalization factor.
    else:
      for col in range (x.shape[1]): # Traverse through all columns of the dataset.
        min_val = np.min(x[:,col])   # Traverse through all rows and get the minimum value of the ith column.
        max_val = np.max(x[:,col])   # Traverse through all rows and get the maximum value of the ith column.
        norm_X[:,col] = (x[:,col]-min_val)/(max_val-min_val) # Normalization. Replace each element of the i-th column by its normalization factor.
      return norm_X
    
    
def shuffle(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  '''
  This function shuffles the dataset.

  Args:
      - a (numpy.ndarray): the normalized dataset.

  Args:
      - b (numpy.ndarray): the array with the corresponding one-hot labels.

  Returns:
      - output (tuple): feature-label pairs of type numpy.ndarray.

  Example:
    >>> shuffle_together(np.random.random((2, 4)), np.random.random((2, 3)))
		(array([[...],[...]]), array([[...],[...]]))
  '''
  print('Running shuffle() function on dataset...')
  random.seed(1)
  indices = np.arange(a.shape[0])
  random.shuffle(indices)	
  new_a = np.empty(a.shape,dtype=a.dtype)
  new_b = np.empty(b.shape,dtype=b.dtype)
  for old_index,new_index in enumerate(indices):
    new_a[new_index] = a[old_index]
    new_b[new_index] = b[old_index]
    output = new_a, new_b
  
  return output
  
  
def preprocessing(file) -> Dict: 
  '''
  This function performs data preprocessing.
  
  Returns:
    - output (dict): test and training datasets of type np.ndarray.
  '''   
  # Cleaning:
  features, labels = clean(file)
  
  # Using only the first two classes:
  print('Using only the first two classes:')
  features = features[:100]
  labels = labels[:100]
  print(f'>>> features.shape = {features.shape}.\n>>> labels.shape = {labels.shape}.\n')
  
  # Normalizing the features:
  features = min_max_norm(features)
  print(f'>>> min = {features.min()}.\n>>> max = {features.max()}.\n')
  
  # One-hot encoding on labels for NN only:
  #labels = one_hot(labels)
  
  # Scaling the labels:
  print('Scaling the labels to [-1,1]:')
  labels = 2 * (labels - 0.5)
  print(f'>>> labels.min() = {labels.min()}.\n>>> labels.max() = {labels.max()}.\n')
  
  # Shuffling the dataset: 
  x, y = shuffle(features,labels)
    
  # Splitting into training and test data:
  print('\nSplitting dataset into test and training:')
  x_train, y_train = x[:75], y[:75]
  x_test, y_test = x[75:], y[75:]
  print(f'>>> x_train.shape = {x_train.shape}.\n>>> y_train.shape={y_train.shape}.\n>>> x_test.shape={x_test.shape}.\n>>> y_test.shape={y_test.shape}.\n')
    
  output = {'x_train':x_train, 'x_test':x_test, 'y_train':y_train, 'y_test':y_test}
  return output

# Testing
if __name__ == '__main__':
  #placeholder = np.random.randint(low=0, high=3, size=5)#.reshape(-1,1)  
  #print(f'{one_hot(placeholder)}')
  preprocessing('iris.txt')