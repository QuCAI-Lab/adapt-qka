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

"""QKA Algorithm"""

###########################################################################
# Quantum circuit:
import pennylane as qml
from pennylane import numpy as np # The Pennylane wrapped NumPy.
#from pennylane.operation import Operation, AnyWires
#from pennylane.optimize import NesterovMomentumOptimizer
#from pennylane.templates import AngleEmbedding

# Dataset 
from .preprocessing import preprocessing
# from preprocessing import preprocessing

# Training
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Real Device
from .ibm_token import TOKEN

# Coupling Map
from qiskit import IBMQ
from qiskit.transpiler import CouplingMap
#from tqdm import trange

# Plotting:
#import matplotlib
#from matplotlib import pyplot as plt
#plt.rcParams.update({'font.size': 10})  # Enlarge matplotlib fonts.
###########################################################################

###########################################################################
#For reproducibility only:
np.random.seed(seed=23)
###########################################################################

def load_ibm():
  hub = 'qhack-event'
  group = 'main'
  project = 'level-1'
  backend_name = "ibmq_lima"
  hgp = f'{hub}/{group}/{project}'
  #print('\n',hgp)
  # Loading account:
  IBMQ.save_account(TOKEN, overwrite=True) 
  # Connect local computer to IBMQ cloud account:
  provider = IBMQ.load_account()
  # Connecting to IBM provider:
  ##provider = IBMQ.get_provider(hub=hub, group=group, project=project)
  #print(f'\nAvailable Providers:\n>>> {IBMQ.providers()}\n')
  #print(f'\nAvailable Backends:\n>>> {provider.backends()}\n')
  return provider

def transpiler(circuit, dev, provider):
  backend = provider.get_backend('ibmq_lima')
  num_qubits = backend.configuration().n_qubits 
  qubit_layout = CouplingMap(getattr(backend.configuration(), 'coupling_map', None) ).reduce([i for i in range(num_qubits)])
  print(f'\nQubit layout:\n{qubit_layout}\n')
  list = []
  adj_M=np.zeros((num_qubits,num_qubits))
  # check if the edge is already exit
  for i in qubit_layout:
    if adj_M[i[1]][i[0]] == 0:
      adj_M[i[0]][i[1]] = 1
      list.append(tuple(i))
  print(f'\nQubit layout list of tuple:\n{list}\n')
  #print(qubit_layout.draw())  
  # transpiled_circuit = qml.transforms.transpile(coupling_map=qubit_layout)(circuit)
  transpiled_circuit = qml.transforms.transpile(coupling_map=list)(circuit)
  transpiled_qnode = qml.QNode(transpiled_circuit, dev)
  # print(qml.draw(transpiled_qnode)())
  return transpiled_qnode


  
class AdaptQKA:
  def __init__(self, data, params=False, real_device=False):
    self.X_train, self.X_test, self.y_train, self.y_test = data.values()
    self.real_device = real_device
    self.nqubits = len(self.X_train[0])
    self.projector = np.zeros((2**self.nqubits, 2**self.nqubits))
    self.projector[0, 0] = 1
    #self.wires = self.dev.wires.tolist()
    if not isinstance(params, np.ndarray):
      self.params = np.random.uniform(0,2*np.pi,(2,self.nqubits), requires_grad=True)
    else:
      self.params=params
    if self.real_device == False:
      print('Running on simulator...\n')
      self.dev = qml.device("default.qubit", wires=self.nqubits, shots=None)
      self.qnode= qml.QNode(self.kernel_ansatz, self.dev)
    else:
      print(f"Running on {self.real_device}...\n")
      provider=load_ibm()
      self.dev = qml.device('qiskit.ibmq', wires=self.nqubits, backend=self.real_device, provider=provider)
      self.qnode = transpiler(self.kernel_ansatz, self.dev, provider)
        
  def fiducial_state_layer(self, lambdas):
    '''
    Fiducial state.
    '''
    for i in range(self.nqubits):
      qml.RY(lambdas[0,i],wires=[i])
    qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=range(self.nqubits), parameters=lambdas[1])

  def encoding_layer(self, x1, x2):
    '''
    Defines the feature map.
    '''
    if len(x1.shape) or len(x2.shape) < 2:
      x1=x1.reshape(1,-1)
      x2=x2.reshape(1,-1)
    # Layer
    [qml.RZ(x1[0,i], wires=[i]) for i in range(self.nqubits)]
    [qml.RX(x1[0,i], wires=[i]) for i in range(self.nqubits)]
    # Layer^(Dagger)
    [qml.adjoint(qml.RX(x2[0,i], wires=[i])) for i in range(self.nqubits)]
    [qml.adjoint(qml.RZ(x2[0,i], wires=[i])) for i in range(self.nqubits)]
  
  def kernel_ansatz(self, x1, x2, params):
    '''
    This method defines the QKA circuit.
    '''  
    self.fiducial_state_layer(params)
    self.encoding_layer(x1, x2)
    #AngleEmbedding(x1, wires=range(self.nqubits))
    #qml.adjoint(AngleEmbedding)(x2, wires=range(self.nqubits))
    qml.adjoint(self.fiducial_state_layer)(params)
    if self.real_device == False:
      return qml.expval(qml.Hermitian(self.projector, wires=range(self.nqubits)))
    else:
      return qml.probs(wires=range(self.nqubits))
    #return qml.probs(wires=wires)
 
  def kernel_value(self, x1, x2, params):
    '''
    Compute the kernel between two data points.
    '''
    print(f'Kernel value between two data points:\n>>> {self.qnode(x1, x2, params)}.\n')

  def show_kernel(self, x1, x2, params):
    '''
    To display the quantum circuit layout.
    '''
    print(f'Kernel ansatz:\n\n{qml.draw(self.qnode)(x1, x2, params)}\n')
    # Exporting circuit image:
    #fig, ax = qml.draw_mpl(qnode)(x1, x2, params)
    
  def kernel_matrix(self, A, B, params):
    '''
    Method to compute the i,j entry of the kernel matrix.
    '''
    length = len(A)
    matrix = [[0 for x in range(length)]for y in range(length)] 

    if self.real_device == False:
      for i in range(length):
        for j in range(i,length):
          matrix[i][j] = (entry := (self.qnode(A[i], B[j], params)))
          if i != j:
            matrix[j][i] = entry
      return np.array(matrix)
    
    else:
      for i in range(self.nqubits):
        for j in range(i, self.nqubits):
          matrix[i][j] = (entry := (self.qnode(A[i], B[j], params)))
          if i != j:
            matrix[j][i] = entry
      return np.array(matrix)
    
  def target_alignment(self, Y, X, kernel_matrix, _lambdas):
      '''
      Kernel-target alignment between kernel and labels.
      '''
      K = kernel_matrix(X, X, _lambdas)
      T = np.outer(Y, Y)
      inner_product = np.sum(K * T) # np.trace(np.dot(K,T)).
      norm = np.sqrt(np.sum(K * K) * np.sum(T * T))
      inner_product = inner_product / norm
      return inner_product
  
  def train(self, epochs, params, threshold=1.0e-3):
    '''
    Training loop.
    Args:
      epochs (int): number of epochs.
      params (np.ndarray): tensor of tunable parameters.
    Returns:
      - params (np.ndarray): the optimized parameters (Euler angles).
    '''   
    print('Optimizing parameters...\n')
    opt = qml.GradientDescentOptimizer(0.2)
    sets = []
    for i in range(epochs):
      subset = np.random.choice(list(range(len(self.X_train))), 4)
      cost = lambda _lambdas: -self.target_alignment(
        self.y_train[subset],
        self.X_train[subset],
        self.kernel_matrix,
        _lambdas)
      
      # Adaptive approach
      '''
      grads = qml.grad(cost)(params)
      maxpos = np.argmax(abs(grads)) 
      minpos = np.argmin(abs(grads)) 
      gatemax=sets[maxpos]
      gatemin=sets[minpos]
      if np.amin(abs(grads)) < threshold:
        sets.remove(gatemin)
        params=np.delete(params, minpos)
      sets.append(gatemax)      
      # Adaptive approach
      '''
      params = opt.step(cost, params)
      # print(f"{i}th's step params: {params}\n")
      current_alignment = self.target_alignment(
        self.y_train,
        self.X_train,
        self.kernel_matrix,
        params)
      print(f'Training step {i+1} ------------> Target Alignment = {current_alignment:.3f}')
    print('\nParameters optimized!')
    return params

  def train_svm(self, params): 
    '''
    Train the SVM classifier.
    '''
    print('\nTrainig SVM...')
    svm = SVC(kernel=lambda X1, X2: self.kernel_matrix(X1, X2, params)).fit(self.X_train, self.y_train)
    print('Trained!\n')
    return svm
  
  def accuracy(self, svm, X, Y_target):
    print('Computing accuracy...')
    acc = 1 - np.count_nonzero(svm.predict(X) - Y_target) / len(Y_target)
    print(f'Accuracy:\n>>> {acc}\n')
    return acc
  
  def prediction(self, svm, test_x, test_y=False):
    '''
    Method for inference.
    
    Returns:
      - acc(float): accuracy.
    '''   
    print(f'Performing QKA inference with {self.nqubits} qubits...')
    predictions = svm.predict(test_x)
    if test_y:
      print(f'Correct label: {test_y[0,0]}')
      accuracy_score(predictions, test_y)    
    print(f'Predicted label: {predictions[0]}\n')
    return predictions[0]
    
if __name__ == '__main__':
  #from preprocessing import preprocessing
  real_device='ibmq_lima'
  data=preprocessing('iris.txt')
  x_train = data['x_train']
  y_train = data['y_train']
  test_x, test_y = data['x_test'], data['y_test']

  # Simulator:
  kernel = AdaptQKA(data)
  # Initialization of parameters:
  params = kernel.params
  # Show kernel value between two datapoints:
  kernel.kernel_value(x_train[0], x_train[1], params)
  # Show kernel circuit:
  #print(data['x_train'][0])
  kernel.show_kernel(x_train[0], x_train[0], params)
  # Show kernel matrix:
  print(f'Kernal matrix between same samples:\n>>> {kernel.kernel_matrix(x_train[:5], x_train[:5], params)}\n')

  # # Training parameters:
  new_params = kernel.train(epochs=2, params=params)

  # Train the SVM:
  svm = kernel.train_svm(new_params)

  # Prediction with one sample:
  kernel.prediction(svm, test_x[0].reshape(1, -1), test_y[0].reshape(1, -1))
  # Show accuracy for the whole training dataset with the optimized parameters:
  print('Accuracy on training dataset:')
  kernel.accuracy(svm, x_train, y_train)
  # Show accuracy for the whole test dataset with the optimized parameters:
  print('Accuracy on test dataset:')
  kernel.accuracy(svm, test_x, test_y)

  # Real device:
  # kernel = AdaptQKA(data, real_device=real_device)
  # # print(f'Kernal matrix between same samples:\n>>> {kernel.kernel_matrix(x_train[:1], x_train[:1], params)}\n')
  # params = kernel.params
  # # print(f'Kernal matrix between same samples:\n>>> {kernel.kernel_matrix(x_train[:1], x_train[:1], params)}\n')
  # params_device = kernel.train(epochs=1, params=params)
  # svm = kernel.train_svm(params_device)
  # kernel.accuracy(svm, x_train, y_train)