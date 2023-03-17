# -*- coding: utf-8 -*-

# @title Copyright 2023.
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

"""QKA Optimization Algorithm"""

###########################################################################
# Quantum circuit:
import pennylane as qml
from pennylane import numpy as np # The Pennylane wrapped NumPy.
# from pennylane.operation import Operation, AnyWires
# from pennylane.optimize import NesterovMomentumOptimizer
# from pennylane.templates import AngleEmbedding

# Dataset
from .preprocessing import preprocessing
# from preprocessing import preprocessing

# Training
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Real Device
from .ibm_token import TOKEN
# from qiskit import IBMQ # The entrypoint qiskit.IBMQ is deprecated.
from qiskit_ibm_provider import IBMProvider

# Coupling Map
from qiskit.transpiler import CouplingMap
# from tqdm import trange

# Function annotation:
from typing import Union, List

# Progress bar
# import time
# from tqdm import tqdm

# Images
# import PIL
# from PIL import Image

# Plotting:
# import matplotlib
# from matplotlib import pyplot as plt
# plt.rcParams.update({'font.size': 10})  # Enlarge matplotlib fonts.
###########################################################################

###########################################################################
# For reproducibility only:
np.random.seed(seed=23)
###########################################################################


def load_ibm(hub='ibm-q', group='open', project='main', backend_name='ibmq_lima'):
  """
  Load IBM account.

  Deprecated:
  IBMQ.save_account(TOKEN, overwrite=True)
  provider = IBMQ.load_account() # Load account from disk and return the default public provider instance.
  IBMQ.providers() # List available providers.
  provider.backends() # List available backends.
  See migration guide at https://qiskit.org/documentation/partners/qiskit_ibm_provider/tutorials/Migration_Guide_from_qiskit-ibmq-provider.html

  Args:
    - hub (str): IBM hub name. Default is 'ibm-q'.
    - group (str): IBM group name. Default is 'open'.
    - project (str): IBM project name. Default is 'main'.
    - backend_name (str): the name of the IBM backend/device/hardware.
  Returns:
    - provider (qiskit_ibm_provider.ibm_provider.IBMProvider): the provider instance.
  """
  hgp = f'{hub}/{group}/{project}'
  print(f'\nLoading IBM account with instance {hgp}...')
  # Saving account:
  IBMProvider.save_account(TOKEN, overwrite=True)
  # Load saved account.
  #provider = IBMProvider()
  # Define a hub/group/project.
  provider = IBMProvider(instance=hgp)
  return provider


def get_qubit_layout(backend_name: str, provider):
  """
  Get the coupling map for a specific IBM device.

  Args:
    - provider (qiskit_ibm_provider.ibm_provider.IBMProvider): the provider instance.
    - backend_name (str): the name of the IBM backend/device/hardware.
  Returns:
    - qubit_layout (list): the device's coupling map.
  """
  backend = provider.get_backend(backend_name)
  num_qubits = backend.configuration().n_qubits
  qubit_layout = CouplingMap(getattr(backend.configuration(), 'coupling_map', None) ).reduce([i for i in range(num_qubits)])
  print(f'\nQubit layout:\n>>> {qubit_layout}')
  print(f'\n{qubit_layout.draw()}') # When using anaconda: "conda install python-graphviz" instead of "pip install graphviz".
  return qubit_layout


def transpiler(circuit, dev, coupling_map: list = None):
  """
  Perform circuit transpilation to match device's connectivity.

  Args:
    - circuit (method): the method to create the quantum circuit.
    - dev (pennylane.devices.default_qubit.DefaultQubit): function that loads the quantum device to construct QNodes.
    - coupling_map (list): the device's coupling map. Default is from ibmq_lima.
  Returns:
    - transpiled_qnode (pennylane.qnode.QNode): the transpiled circuit matching the provided backend coupling map.
  """
  if not coupling_map:
    print('\nTranspiling the quantum circuit to match ibmq_lima default coupling map...')
    coupling_map=[[0, 1], [1, 0], [1, 2], [1, 3], [2, 1], [3, 1], [3, 4], [4, 3]]
  else:
    print('\nTranspiling the quantum circuit to match the provided coupling map...')
  reduced_layout = []
  for entry in coupling_map:
    if (entry[1], entry[0]) not in reduced_layout:
      reduced_layout.append((entry[0], entry[1]))
  transpiled_circuit = qml.transforms.transpile(coupling_map=reduced_layout)(circuit)
  transpiled_qnode = qml.QNode(transpiled_circuit, dev)
  return transpiled_qnode


class AdaptQKA:
  def __init__(self, data: np.ndarray, params: np.ndarray = None, real_device: str = None, gates: np.ndarray = None):
    self.X_train, self.X_test, self.y_train, self.y_test = data.values()
    self.real_device = real_device
    self.nqubits = len(self.X_train[0])
    self.projector = np.zeros((2**self.nqubits, 2**self.nqubits))
    self.projector[0, 0] = 1
    self.gates = gates
    # self.wires = self.dev.wires.tolist()
    if not isinstance(params, np.ndarray):
      self.params = np.random.uniform(0,2*np.pi,(2,self.nqubits), requires_grad=True)
    else:
      self.params=params
    self.dev = qml.device("default.qubit", wires=self.nqubits, shots=None)
    self.qnode = qml.QNode(self.kernel_ansatz, self.dev, interface="autograd")
    self.show_kernel(self.X_train[0], self.X_train[0], self.params)

  def fiducial_state_layer(self, lambdas: np.ndarray, gates: Union[np.ndarray, List] = None):
    """
    Fiducial state layer with adaptive gates.

    Args:
      - lambdas (np.ndarray): tensor of tunable parameters (euler angles).
      - gates (np.ndarray or list): tensor or list with initial gates.
    """
    if not self.gates:
      for i in range(self.nqubits):
        qml.RY(lambdas[0, i], wires=[i])
      qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=range(self.nqubits), parameters=lambdas[1])
    else:
      for gate in gates:
        break

  def encoding_layer(self, x1, x2):
    """
    Defines the feature map.

    Args:
      - x1 (np.ndarray): the first sample/data point.
      - x2 (np.ndarray): the second sample/data point.
    """
    if len(x1.shape) or len(x2.shape) < 2:
      x1=x1.reshape(1,-1)
      x2=x2.reshape(1,-1)
    # Layer
    [qml.RZ(x1[0,i], wires=[i]) for i in range(self.nqubits)]
    [qml.RX(x1[0,i], wires=[i]) for i in range(self.nqubits)]
    # Layer^(Dagger)
    [qml.adjoint(qml.RX(x2[0,i], wires=[i])) for i in range(self.nqubits)]
    [qml.adjoint(qml.RZ(x2[0,i], wires=[i])) for i in range(self.nqubits)]

  def kernel_ansatz(self, x1, x2, params: np.ndarray, gates: Union[np.ndarray, List] = None):
    """
    This method defines the QKA circuit.

    Args:
      - x1 (np.ndarray): the first sample/data point.
      - x2 (np.ndarray): the second sample/data point.
      - params (np.ndarray): tensor of tunable parameters (euler angles).
      - gates (np.ndarray or list): tensor or list with the adaptive gates. Default value is None.
    Returns:
      - probs (pennylane.measurements.probs.ProbabilityMP): probability distribution.
    """
    self.fiducial_state_layer(params)
    self.encoding_layer(x1, x2)
    qml.adjoint(self.fiducial_state_layer)(params)
    #return qml.expval(qml.Hermitian(self.projector, wires=range(self.nqubits)))
    #return qml.probs(wires=self.dev.wires.tolist())
    return qml.probs(wires=list(range(self.nqubits)))

  def kernel_value(self, x1, x2, params: np.ndarray = None, gates: Union[np.ndarray, List] = None):
    """
    Compute the kernel between two data points.

    Args:
      - x1 (np.ndarray): the first sample/data point.
      - x2 (np.ndarray): the second sample/data point.
      - params (np.ndarray): tensor of tunable parameters.
      - gates (np.ndarray or list): the adaptive gates of the quantum circuit.
    """
    if isinstance(params, type(None)):
      params=self.params
    print(f'\nKernel value between the first two data points:\n>>> {self.qnode(x1, x2, params, gates)}')

  def show_kernel(self, x1, x2, params: np.ndarray, gates: Union[np.ndarray, List] = None, message='Kernel Ansatz:'):
    """
    To display the quantum circuit layout.

    Args:
      - params (np.ndarray): tensor of tunable parameters.
      - gates (np.ndarray or list): the adaptive gates of the quantum circuit.
    """
    if isinstance(params, type(None)):
      params=self.params
    print(f'\n{message}\n\n{qml.draw(self.qnode)(x1, x2, params, gates)}')
    # Exporting circuit image:
    #fig, ax = qml.draw_mpl(qnode)(x1, x2, params)

  def kernel_matrix(self, A, B, params: np.ndarray = None, gates=None):
    """
    Method to compute the i,j entry of the kernel matrix.
    """
    if isinstance(params, type(None)):
      params=self.params
    length = len(A)
    matrix = [[0 for x in range(length)] for y in range(length)]
    for i in range(length):
      for j in range(i, length):
        matrix[i][j] = (entry := (self.qnode(A[i], B[j], params, gates)[0]))
        if i != j:
          matrix[j][i] = entry
    return np.array(matrix)

  def target_alignment(self, Y, X, kernel_matrix, _lambdas: np.ndarray, gates:Union[np.ndarray, List] = None):
    """
    Kernel-target alignment between kernel and labels.
    """
    K = kernel_matrix(X, X, _lambdas, gates)
    T = np.outer(Y, Y)
    inner_product = np.sum(K * T) # np.trace(np.dot(K,T)).
    norm = np.sqrt(np.sum(K * K) * np.sum(T * T))
    inner_product = inner_product / norm
    return inner_product

  def train(self, epochs: int, params: np.ndarray = None, threshold: float=1.0e-4, coupling_map: list=None, gates:Union[np.ndarray, List] = None):
    """
    Training loop.

    Args:
      - epochs (int): number of epochs.
      - params (np.ndarray): tensor of tunable parameters.
      - threshold (float): Default is 1.0e-4.
      - coupling_map (list): a device's coupling map for transpilation of the circuit ansatz. Default is None.
      - gates (np.ndarray or list): the adaptive gates of the quantum circuit. Default is None.
    Returns:
      - params (np.ndarray): the optimized parameters (Euler angles).
      - gates (np.ndarray or list): the new list of gates.
    """
    if isinstance(params, type(None)):
      params = self.params
    if not self.real_device:
      self.dev = qml.device("qiskit.aer", wires=self.nqubits)
      self.qnode = transpiler(self.kernel_ansatz, self.dev, coupling_map)
      self.show_kernel(self.X_train[0], self.X_train[0], self.params, message='Transpiled Kernel Ansatz:')
      print('\nOptimizing quantum circuit parameters on pennylane default simulator...')
    else:
      load_ibm()
      print(f'\nOptimizing quantum circuit parameters on {self.real_device}...')
      self.dev = qml.device('qiskit.ibmq', wires=self.nqubits, backend=self.real_device)
      self.qnode= qml.QNode(self.kernel_ansatz, self.dev, interface="autograd")
    if not gates:
      gates=[]
    opt = qml.GradientDescentOptimizer(0.2)
    for i in range(epochs):
      #time.sleep(1)
      subset = np.random.choice(list(range(len(self.X_train))), 4)
      cost = lambda _lambdas: -self.target_alignment(
        self.y_train[subset],
        self.X_train[subset],
        self.kernel_matrix,
        _lambdas)
      ### Adaptive approach ###
      '''
      grads = qml.grad(cost)(params)
      maxpos = np.argmax(abs(grads))
      minpos = np.argmin(abs(grads))
      gatemax=gates[maxpos]
      gatemin=gates[minpos]
      if np.amin(abs(grads)) < threshold:
        gates.remove(gatemin)
        params=np.delete(params, minpos)
      gates.append(gatemax)     
      # Adaptive approach
      '''
      ### End of Adaptive approach ###
      params = opt.step(cost, params)
      current_alignment = self.target_alignment(
        self.y_train,
        self.X_train,
        self.kernel_matrix,
        params)
      print(f'\nTraining step {i+1} ------------> Target Alignment = {current_alignment:.3f}')
    print('\nParameters optimized!')
    return params, gates

  def train_svm(self, params: np.ndarray):
    """
    Train the SVM classifier.

    Args:
      - params (np.ndarray): tensor of tunable parameters.
    Returns:
      - svm (sklearn.svm._classes.SVC): the support vector machine classifier.
    """
    print('\nTrainig SVM...')
    svm = SVC(kernel=lambda x1, x2: self.kernel_matrix(x1, x2, params)).fit(self.X_train, self.y_train)
    print('Trained!')
    return svm

  def accuracy(self, svm, x, y_target):
    """
    Method to compute the accuracy.

    Args:
      - svm (sklearn.svm._classes.SVC): the support vector machine classifier.
      - X (np.ndarray): samples.
      - Y_target (np.ndarray): labels.
    Returns:
      - acc (float): accuracy.
    """
    print('\nComputing accuracy...')
    acc = 1 - np.count_nonzero(svm.predict(x) - y_target) / len(y_target)
    print(f'Accuracy: {acc}')
    return acc

  def prediction(self, svm, test_x, test_y=False):
    """
    Method for inference.

    Args:
      - svm (sklearn.svm._classes.SVC): the support vector machine classifier.
      - test_x (np.ndarray): test sample, a tensor of features.
      - test_y (np.ndarray): test label. Default is False.
    Returns:
      - prediction (float): predicted label for input sample.
    """
    print(f'\nPerforming QKA inference with {self.nqubits} qubits...')
    predictions = svm.predict(test_x)
    if test_y:
      print(f'Correct label: {test_y[0,0]}')
      accuracy_score(predictions, test_y)
    print(f'Predicted label: {predictions[0]}')
    return predictions[0]

if __name__ == '__main__':
  ############## Data preprocessing: ##############
  dataset=preprocessing('iris.txt')
  x_train = dataset['x_train']
  y_train = dataset['y_train']
  x_test, y_test = dataset['x_test'], dataset['y_test']

  #print(f'\n{(x_train[:1][0] == x_train[0]).all()}') # >>> True

  ############## Simulator: ##############
  kernel = AdaptQKA(data=dataset, params=None)
  # Show kernel value between two datapoints:
  kernel.kernel_value(x_train[0], x_train[1])
  # Show kernel matrix:
  print(f'\nKernal matrix between same samples:\n>>> {kernel.kernel_matrix(x_train[:1], x_train[:1])}')

  # Training parameters with circuit transpilation using custom coupling map:
  provider=load_ibm()
  qubit_layout = get_qubit_layout('ibmq_manila', provider)

  # Training parameters with circuit transpilation using default coupling map:
  new_params, gates = kernel.train(epochs=1, threshold=1.0e-5, coupling_map=qubit_layout)

  # Show current quantum circuit:
  kernel.show_kernel(x_train[0], x_train[0], new_params, gates, message='Current circuit with optimized parameters:')
  # Train the SVM:
  svm = kernel.train_svm(new_params)

  # Prediction with one sample:
  kernel.prediction(svm, x_test[0], y_test[0])
  # Show accuracy for the whole training dataset with the optimized parameters:
  print('\nAccuracy on training dataset:')
  kernel.accuracy(svm, x_train, y_train)
  # Show accuracy for the whole test dataset with the optimized parameters:
  print('\nAccuracy on test dataset:')
  kernel.accuracy(svm, x_test, y_test)
  '''
  ############## Real device: ##############
  # Define the kernel for the real quantum device:
  kernel = AdaptQKA(data, real_device='ibmq_lima')
  # Training parameters:
  params_device, gates = kernel.train(epochs=1, params=params, threshold=1.0e-5)
  # Show current quantum circuit:
  kernel.show_kernel(x_train[0], x_train[0], params_device, gates, message='Current circuit with optimized parameters:')
  # Train the SVM:
  svm = kernel.train_svm(params_device)
  # Show accuracy for the whole training dataset with the optimized parameters:
  print('\nAccuracy on training dataset:')
  kernel.accuracy(svm, x_train, y_train)
  '''
