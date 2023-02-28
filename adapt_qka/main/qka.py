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

# Training
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Real Device
from ibm_token import TOKEN

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
  IBMQ.load_account()
  IBMQ.save_account(TOKEN)
  provider = IBMQ.load_account()
  #print(IBMQ.providers())
  #print(provider.backends())
  return provider

class AdaptQKA:
  def __init__(self, data, params=False, real_device=False):
    self.X_train, self.X_test, self.y_train, self.y_test = data.values()
    self.real_device = real_device
    self.nqubits = len(self.X_train[0])
    if not isinstance(params, np.ndarray):
      self.params = np.random.uniform(0,2*np.pi,(2,self.nqubits), requires_grad=True)
    else:
      self.params=params
    self.projector = np.zeros((2**self.nqubits, 2**self.nqubits))
    self.projector[0, 0] = 1
    self.dev = qml.device("default.qubit", wires=self.nqubits, shots=None)
    #self.wires = self.dev.wires.tolist()

  def transpiler(self):
    provider = load_ibm()
    backend = provider.get_backend('ibmq_lima')
    num_qubits = backend.configuration().n_qubits 
    qubit_layout = CouplingMap(getattr(backend.configuration(), 'coupling_map', None) ).reduce([i for i in range(num_qubits)])
    print(f'\nQubit layout:\n{qubit_layout}\n')
    #print(qubit_layout.draw())  
    circuit = self.kernel_ansatz
    transpiled_circuit = qml.transforms.transpile(coupling_map=qubit_layout)(circuit)
    transpiled_qnode = qml.QNode(transpiled_circuit, self.dev)
    return transpiled_qnode

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
    return qml.expval(qml.Hermitian(self.projector, wires=range(self.nqubits)))
    #return qml.probs(wires=wires)
 
  def kernel_value(self, x1, x2, params):
    print(f'Kernel value between two datapoints:\n>>> {qml.QNode(self.kernel_ansatz, self.dev)(x1, x2, params)}.\n')

  def show_kernel(self, x1, x2, params):
    '''
    To display the quantum circuit layout.
    '''
    dev = qml.device("default.qubit", wires=4)
    qnode = qml.QNode(self.kernel_ansatz, dev)
    print(f'Kernel ansatz:\n\n{qml.draw(qnode)(x1, x2, params)}\n')
    
    # Exporting circuit image:
    #fig, ax = qml.draw_mpl(qnode)(x1, x2, params)
    
  def kernel_matrix(self, A, B, params):
    '''
    Method to compute the i,j entry of the kernel matrix.
    '''
    if self.real_device == False:
      #print('Running on simulator...\n')
      qnode= qml.QNode(self.kernel_ansatz, self.dev)
    else:
      dev = qml.device('qiskit.ibmq', wires=self.nqubits, backend=self.real_device)
      #print(f"Running on {self.real_device}...\n")
      qnode = self.transpiler(data)
    return np.array([[qnode(a, b, params) for b in B] for a in A])

  def target_alignment(self, Y, kernel_matrix):
      '''
      Kernel-target alignment between kernel and labels.
      '''
      K = kernel_matrix
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
          self.kernel_matrix(self.X_train[subset], self.X_train[subset], _lambdas),
      )
      
      # Adaptive approach
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
      
      params = opt.step(cost, params)
      # print(f"{i}th's step params: {params}\n")
      current_alignment = self.target_alignment(
              self.y_train,
              self.kernel_matrix(self.X_train, self.X_train, params),
          )
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
    print(f'Accuracy:\n>>> {1 - np.count_nonzero(svm.predict(X) - Y_target) / len(Y_target)}\n')
  
  def prediction(self, svm, test_x, test_y=False):
    '''
    Method for inference.
    
    Returns:
      - acc(float): accuracy.
    '''   
    print(f'Performing QKA inference with {self.nqubits} qubits...')
    predictions = svm.predict(test_x)
    if test_y:
      print(f'Correct labels: {test_y[0]}')
      accuracy_score(predictions, test_y)    
    print(f'Predicted label: {predictions[0]}\n')
    return predictions
    
if __name__ == '__main__':
  from preprocessing import preprocessing
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
  print(f'Kernal matrix between same samples:\n>>> {kernel.kernel_matrix(x_train[:1], x_train[:1], params)}\n')

  # Training parameters:
  new_params = kernel.train(epochs=10, params=params)
  #print(f'Trained parameter:\n{new_params}')

  # Train the SVM:
  svm = kernel.train_svm(new_params)

  # Prediction with one sample:
  kernel.prediction(svm, test_x[0].reshape(1, -1))
  # Show accuracy for the whole training dataset with the optimized parameters:
  print('Accuracy for training data:')
  kernel.accuracy(svm, x_train, y_train)
  # Show accuracy for the whole test dataset with the optimized parameters:
  print('Accuracy for test data:')
  kernel.accuracy(svm, test_x, test_y)

  # Real device:
  kernel = AdaptQKA(data, real_device)
  #params_device = kernel.train(epochs=1, params=params)