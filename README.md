<!-- Shields: -->
[![Python](https://img.shields.io/badge/Python-3.8.8-informational)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-%E2%89%A5%200.28.0-6133BD)](https://pennylane.ai/)
[![License](https://img.shields.io/github/license/QuCAI-Lab/adapt-qka.svg?logo=CreativeCommons&style=flat-square)](https://github.com/QuCAI-Lab/adapt-qka/blob/dev/LICENSE.md)
[![Contributions](https://img.shields.io/badge/contributions-welcome-orange?style=flat-square)](https://github.com/QuCAI-Lab/adapt-qka/pulls)

<!-- Logo: -->
<div align="center">
  <a href="https://qucai-lab.github.io/">
    <img src="https://github.com/QuCAI-Lab/qucai-lab.github.io/blob/main/assets/QuCAI-Lab.png" height="500" width="500" alt="Logo">
  </a>
</div>

<!-- Title: -->
<div align="center">
  <h1><a href="https://github.com/XanaduAI/QHack2023"> QHack2023 Open Hackathon </a></h1>
  <h2> Adaptive Quantum Kernel Alignment for data Classification </h2>
</div>
<br> 
<br> 


<!-- Team: -->
<div align="center">
  <b>
    Authors: <a target="_blank" href="https://github.com/camponogaraviera">¹²Lucas Camponogara Viera</a>, 
    <a target="_blank" href="https://github.com/wormyu">²陳宏宇</a>. 
  </b>
<!-- Institution: -->
<br>
<b><a target="_blank" href="https://github.com/QuCAI-Lab">¹QuCAI-Lab, Taipei, Taiwan</a></b>.
<br>
<b><a target="_blank" href="https://quantum.ntu.edu.tw/?page_id=275">²IBM Quantum Hub at NTU, Taipei, Taiwan</a></b>.<br>
</div>


<!-- Dependencies: -->
# Dependencies
<a href="https://www.python.org/" target="_blank" rel="noopener noreferrer"><img height="27" src="https://www.python.org/static/img/python-logo.png"></a>
<a href="https://numpy.org/" target="_blank" rel="noopener noreferrer"><img height="27" src="https://numpy.org/images/logo.svg"></a>
<a href="https://matplotlib.org" target="_blank" rel="noopener noreferrer"><img height="27" src="https://matplotlib.org/_static/images/logo2.svg"></a>
<a href="https://pennylane.ai/" target="_blank" rel="noopener noreferrer"><img height="27" src="https://pennylane.ai/img/logo.png"></a>
<a target="_blank" href="https://www.tensorflow.org/"><img height="30" src="https://www.gstatic.com/devrel-devsite/prod/v2484c9574f819dcf3d7ffae39fb3001f4498b2ece38cec22517931d550e19e7d/tensorflow/images/lockup.svg" /></a>
<a target="_blank" href="https://keras.io/"><img height="30" src="https://keras.io/img/logo.png" /></a>
<a href="https://pytorch.org/"  target="_blank" rel="noopener noreferrer"><img height="30" src="https://pytorch.org/assets/images/pytorch-logo.png"></a> 
<a target="_blank" href="https://scikit-learn.org/stable/"><img height="30" src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" /></a>
<a target="_blank" href="https://pandas.pydata.org/docs/"><img height="30" src="https://pandas.pydata.org/docs/_static/pandas.svg" /></a>
<br>
<br>
  
For specific versions, see the [requirements.txt](requirements.txt) file.


<!-- Results: -->
# Results 

View source code [here](https://github.com/QuCAI-Lab/adapt-qka/tree/dev/adapt_qka). To know about the step-by-step implementation, one can resort to the [supplementary](supplementary.ipynb) material for a skill reaffirming theoretical background. Do not hesitate to open an issue in the [issue tracker](https://github.com/QuCAI-Lab/adapt-qka/issues).

```bash
Running clean() function on iris.txt file:
>>> features.shape = (150, 4).
>>> labels.shape = (150,)

Using only the first two classes:
>>> features.shape = (100, 4).
>>> labels.shape = (100,).

Running min_max_norm() function on features:
>>> min = 0.0.
>>> max = 1.0.

Scaling the labels to [-1,1]:
>>> labels.min() = -1.0.
>>> labels.max() = 1.0.

Running shuffle() function on dataset...

Splitting dataset into test and training:
>>> x_train.shape = (75, 4).
>>> y_train.shape=(75,).
>>> x_test.shape=(25, 4).
>>> y_test.shape=(25,).

Kernel Ansatz:

0: ──RY(3.25)─╭●────────────────────────────╭RZ(2.47)──RY(3.88)──RZ(0.44)──RX(0.44)──RX(0.44)†
1: ──RY(5.95)─╰RZ(1.39)─╭●──────────────────│──────────RY(2.59)──RZ(0.17)──RX(0.17)──RX(0.17)†
2: ──RY(4.81)───────────╰RZ(4.31)─╭●────────│──────────RY(0.02)──RZ(0.66)──RX(0.66)──RX(0.66)†
3: ──RY(1.77)─────────────────────╰RZ(1.05)─╰●─────────RY(5.55)──RZ(0.53)──RX(0.53)──RX(0.53)†

───RZ(0.44)†──RY(3.88)†─╭RZ(2.47)†───────────────────────╭●──────────RY(3.25)†─┤ ╭Probs
───RZ(0.17)†──RY(2.59)†─│─────────────────────╭●─────────╰RZ(1.39)†──RY(5.95)†─┤ ├Probs
───RZ(0.66)†──RY(0.02)†─│──────────╭●─────────╰RZ(4.31)†──RY(4.81)†────────────┤ ├Probs
───RZ(0.53)†──RY(5.55)†─╰●─────────╰RZ(1.05)†──RY(1.77)†───────────────────────┤ ╰Probs

Kernel value:
>>> 0.7860882708330421

Kernel matrix:
>>> [[1.]]

Transpiling the quantum circuit to match the default coupling map ibmq_lima...

Transpiled Kernel Ansatz:

0: ──RY(3.25)─╭●──────────────────────────────────╭RZ(2.47)──RY(3.88)──RZ(0.44)───RX(0.44)─
1: ──RY(5.95)─╰RZ(1.39)─╭●────────╭SWAP─╭RZ(1.05)─╰●─────────RY(5.55)──RZ(0.53)───RX(0.53)─
2: ──RY(4.81)───────────╰RZ(4.31)─│─────╰●─────────RY(0.02)──RZ(0.66)──RX(0.66)───RX(-0.66)
3: ──RY(1.77)─────────────────────╰SWAP──RY(2.59)──RZ(0.17)──RX(0.17)──RX(-0.17)──RZ(-0.17)

───RX(-0.44)──RZ(-0.44)──RY(-3.88)─╭RZ(-2.47)───────────────────────────────────╭●────────
───RX(-0.53)──RZ(-0.53)──RY(-5.55)─╰●─────────╭RZ(-1.05)─╭SWAP─╭RZ(-4.31)─╭SWAP─╰RZ(-1.39)
───RZ(-0.66)──RY(-0.02)───────────────────────╰●─────────╰SWAP─│──────────│──────RY(-1.77)
───RY(-2.59)───────────────────────────────────────────────────╰●─────────╰SWAP──RY(-4.81)

───RY(-3.25)─┤ ╭Probs
───RY(-5.95)─┤ ├Probs
─────────────┤ ├Probs
─────────────┤ ╰Probs

Optimizing quantum circuit parameters on pennylane default simulator...

Training step 1 ------------> Target Alignment = 0.137

Parameters optimized!

Current circuit with optimized parameters:

0: ──RY(3.25)─╭●──────────────────────────────────╭RZ(2.47)──RY(3.88)──RZ(0.44)───RX(0.44)─
1: ──RY(5.95)─╰RZ(1.39)─╭●────────╭SWAP─╭RZ(1.06)─╰●─────────RY(5.55)──RZ(0.53)───RX(0.53)─
2: ──RY(4.80)───────────╰RZ(4.31)─│─────╰●─────────RY(0.01)──RY(4.81)──RZ(0.66)───RX(0.66)─
3: ──RY(1.78)─────────────────────╰SWAP──RY(2.59)──RZ(0.17)──RX(0.17)──RX(-0.17)──RZ(-0.17)

───RX(-0.44)──RZ(-0.44)──RY(-3.88)─╭RZ(-2.47)───────────────────────────────────╭●────────
───RX(-0.53)──RZ(-0.53)──RY(-5.55)─╰●─────────╭RZ(-1.06)─╭SWAP─╭RZ(-4.31)─╭SWAP─╰RZ(-1.39)
───RX(-0.66)──RZ(-0.66)──RY(-4.81)──RY(-0.01)─╰●─────────╰SWAP─│──────────│──────RY(-1.78)
───RY(-2.59)───────────────────────────────────────────────────╰●─────────╰SWAP──RY(-4.80)

───RY(-3.25)─┤ ╭Probs
───RY(-5.95)─┤ ├Probs
─────────────┤ ├Probs
─────────────┤ ╰Probs

Trainig SVM...
Trained!

Performing QKA inference with 4 qubits...
Correct label: versicolor
Predicted label: versicolor

Accuracy on training dataset:

Computing accuracy...
Accuracy: 1.0

Accuracy on test dataset:

Computing accuracy...
Accuracy: 1.0

Kernel Ansatz:

0: ──RY(2.63)─╭●────────────────────────────╭RZ(4.19)──RY(4.75)──RZ(0.44)──RX(0.44)──RX(0.44)†
1: ──RY(4.28)─╰RZ(2.48)─╭●──────────────────│──────────RY(5.24)──RZ(0.17)──RX(0.17)──RX(0.17)†
2: ──RY(4.73)───────────╰RZ(3.01)─╭●────────│──────────RY(1.08)──RZ(0.66)──RX(0.66)──RX(0.66)†
3: ──RY(3.64)─────────────────────╰RZ(2.46)─╰●─────────RY(1.33)──RZ(0.53)──RX(0.53)──RX(0.53)†

───RZ(0.44)†──RY(4.75)†─╭RZ(4.19)†───────────────────────╭●──────────RY(2.63)†─┤ ╭Probs
───RZ(0.17)†──RY(5.24)†─│─────────────────────╭●─────────╰RZ(2.48)†──RY(4.28)†─┤ ├Probs
───RZ(0.66)†──RY(1.08)†─│──────────╭●─────────╰RZ(3.01)†──RY(4.73)†────────────┤ ├Probs
───RZ(0.53)†──RY(1.33)†─╰●─────────╰RZ(2.46)†──RY(3.64)†───────────────────────┤ ╰Probs

Loading IBM account with instance ibm-q/open/main...

Optimizing quantum circuit parameters on ibmq_lima real hardware...
```

<!-- Description: -->
# Project Description 

In this work, we extend the quantum kernel alignment (QKA) first proposed in [[1](#1)] to demonstrate that image classification can be realized by optimizing the parameters of the fiducial state while varying its quantum circuit structure using the adaptive circuit approach. In addition, a recurrent neural network (RNN) could also be used for parameter initialization. The variational quantum kernel of our Quantum Kernel Training (QKT) algorithm (figure 1) has two main layers:

1. The V layer a.k.a fiducial state has parameterized single qubit and two-qubit entangler CPhase gates. The parameters (Euler angles) are optimized via gradient descent while the structure of the layer changes adaptively \[[2](#2)] according to the gradient of the gates computed with the parameter shift rule. At each training iteration, the gate with the highest gradient will be copied to the circuit, while a gate will be removed if its gradient is smaller than a pre-defined threshold. 
2. The D layer is used to encode the data points. 

<div align="center">
    <img src="https://github.com/QuCAI-Lab/adapt-qka/blob/dev/assets/circuit.png" height="350" width="450" alt="Figure 1">
    <p> Figure 1: the proposed circuit structure.</p>
</div>

## Introduction

Classical [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning) (CML) is a suite of inter-related and data-driven algorithms that parse data to learn its meaningful patterns of interest throughout an iterative approach enabling future predictions. A popular subset of CML is [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning) that uses Artificial Neural Networks playing the role of [universal function approximators](https://en.wikipedia.org/wiki/Universal_approximation_theorem) for [representation learning](https://en.wikipedia.org/wiki/Feature_learning). 

"A computer program is said to learn from experience $\mathcal{E}$ with respect to some class of tasks $\mathcal{T}$ and performance measure $\mathcal{P}$, if its performance at tasks in $\mathcal{T}$, as measured by $\mathcal{P}$, improves with experience $\mathcal{E}.$" —[[Tom M. Mitchell, 1997, pg.2](https://books.google.com.tw/books/about/Machine_Learning.html?id=EoYBngEACAAJ&redir_esc=y)]. 

On the other hand, Quantum Machine Learning (QML) is the interplay between quantum computing and classical machine learning. From one side, quantum computing has the potential to speed up certain classical optimization tasks using the quantum hardware as a hardware accelerator, while quantum circuits can be trained just like classical neural networks using gradient descent via [parameter shift rule](https://pennylane.ai/qml/glossary/parameter_shift.html). From the other side, classical machine learning can be used to power quantum computing tasks, such as in the design of new error mitigation schemes (see [Mitiq](https://mitiq.readthedocs.io/en/stable/examples/examples.html)), in the design of new quantum circuits for quantum optimization, and finding an optimal quantum circuit ansatz (see [this project](https://github.com/QuCAI-Lab/qhack2022-hackeinberg-project)) for quantum chemistry simulations. 

A quantum computer (QC) only outperforms a classical computer for specific classes of problems, i.e, QCs are not general purpose computers. Moreover, for those classes of problems, a quantum algorithm will only outstrip or gain quantum advantage over a classical algorithm if the principle of interference is implemented. In optics, quantum interference is the phenomenon whereby wave sources constructively or destructively interfere depending on their path difference. In quantum computing, quantum interference causes wrong answers of a computation to destructively interfere and right answers to constructively interfere. In this sense, eigenstates correspoding to wrong answers will have lower probability amplitudes of occurrence while eigenstates correspoding to right answers will have higher probability amplitudes. The development of quantum algorithms is a challenge because it is difficult to grasp an intuition on how to implement quantum interference using entangling gates. 

Quantum algorithms can be classified in two classes: longer-term algorithms and near-term algorithms. The former requires quantum computers which are fault-tolerant, i.e, allowing for a platform with millions of qubits. The latter refers to algorithms tailored to the current noisy intermediate-scale quantum (NISQ) \[[3](#3)] devices that are not fault-tolerant and have superconducting circuit-based platforms with only a few hundred qubits. Current NISQ devices resemble special-purpose classical hardware such as Application-Specific Integrated Circuits (ASICs) and Field-Programmable Gate Arrays (FPGAs), in the sense that NISQ devices are tailored to run specific quantum algorithms using quantum logic gates programmable with a hardware-specific programming language. 

The primer examples of longer-term algorithms to have shown a quantum speed-up over their classical counterparts were devised by Peter Shor \[[4](#4)]\[[5](#5)] between 1994-1997 and Lov Grover \[[6](#6)] in 1996. Shor's quantum algorithm is based on the quantum Fourier transform for finding the prime factors of an integer (RSA decryption) and solving the discrete logarithm problem. While the best classical algorithm used to factor an $n$-digit number has exponential-time complexity $\mathcal{O}(2^n)$, Shor's algorithm has polynomial-time complexity $\mathcal{O}(n^2)$ and provides a polynomial speedup in time. In Grover's quantum searching algorithm, while the time-complexity of the best classical search algorithm to search an element from a list of $N$ elements scales linearly as order $\mathcal{O}(N)$, Grover’s quantum search algorithm scales with order $\mathcal{O}(\sqrt{N})$, i.e, it requires only $\sqrt{N}$ operations. Grover’s search algorithm performs a quantum searching database with a quadratic speedup in time. A flagship among near-term quantum algorithms is the hybrid quantum-classical Variational Quantum Eigensolver (VQE) \[[7](#7)]\[[8](#8)], an application of the Ritz variational principle used to find the ground state corresponding to the lowest energy level ($E_o$) of a chemical molecule. The ground state of a molecule provides access to information of molecular reaction rates ($r \propto e^{\mathcal{O}(\Delta E_0)}$) and molecular structure used to drive the design of new drugs, materials and electric batteries.

As an alternative to the Variational Quantum Classifier (VQC), the Quantum Kernel Estimator (QKE) algorithm uses the kernel trick to implicitly compute the feature map without ever computing the feature vectors. Quantum kernels are believed to provide two main quantum advantages over their classical counterpart: superpolynomial speed up \[[9](#9)] and better feature extraction from input data. An extension of the QKE is the Quantum Kernel Alignment (QKA) \[[1](#1)]. In this new setting, the $ij$-th entry of the quantum kernel function in the explicit form is computed in the same way as the QKE (see the [supplementary material](supplementary.ipynb) for derivation), by taking the Hilbert-Schmidt inner product of two quantum feature states parameterized by distinct data points $\vec{x}_i$ and $\vec{x}_j$ from a dataset $\mathcal{X}$:

$$K_{ij}(\vec{x}_i, \vec{x}_j) = |\langle\Phi(\vec{x}_i)|\Phi(\vec{x}_j)\rangle_{HS}|^2 = |\langle 0^{\otimes n}|U^{\dagger}_{\Phi}(\vec{x}_i)U_{\Phi}(\vec{x}_j)|0^{\otimes n}\rangle|^2 = \underbrace{|\langle0^{\otimes n}|V^{\dagger}(\lambda)}_{\text{fiducial state}} \underbrace{D^{\dagger}_{\Phi}(\vec{x}_i)D_{\Phi}(\vec{x}_j)}_{\text{feature map}}\underbrace{V(\lambda)|0^{\otimes n}\rangle}_{\text{fiducial state}}|^2 \in \mathbb{R}.$$

Note that $V(\lambda)$ is the unitary representation for the fiducial state containing the entangler gates, and $D_{\Phi}(\vec{x}_i)$ is the unitary representation of the quantum feature map for data encoding.

The output of any quantum circuit is a quantum state that can be measured. This measurement can be an expectation value measurement, a projective measurement (a.k.a von Neumann measurement) for probabilities, or a POVM measurement. The most general expectation value measurement can be written as: 

$$\langle\hat{\mathcal{O}}\rangle \doteq \langle \psi| \hat{\mathcal{O}} |\psi\rangle = \langle 0^{\otimes n}|\hat{U}^{\dagger}W^{\dagger} \hat{\mathcal{O}} W\hat{U}|0^{\otimes n}\rangle \in \mathbb{R},$$
where $W$ denotes the change of basis circuit used to rotate from the eigenbasis of the observable into the computational basis when the observable is not diagonalized in the z-basis. In the variational class of algorithms (VQE, SVM, Quantum Kernels...), the quantum observable $\hat{\mathcal{O}} \in \mathcal{H}$ can encode quantities such as the electronic structure of a molecule or the cost function for an optimization problem.

## Approach

In this work, the proposed quantum loss (cost) function for the quantum circuit is the `kernel-target alignment` for `multiclass classification` with the iris dataset:

$$KA(K_1,K_2) = \frac{\text{Tr}(K_1K_2)}{\sqrt{\text{Tr}(K_1^2)\text{Tr}(K_2^2)}}.$$

The emergent field of QML includes the development of hybrid quantum-classical algorithms, and when designing such algorithms it is helpful to have a set of rule of thumb as outlined below.

- **On the classical side:**
  1. `Ockham's razor:` when there are two competing theories that make exactly the same predictions, the simpler one is better. In model complexity, the more complex the model, the more prone the model is to overfitting as the size of the dataset decreases. Therefore, one should always start with a baseline model.
  2. `Learning principle:` random features (noise) cannot be learned. In a data-driven approach, the dataset must share a pattern of meaningful representation.
  3. `Hold-out set:` it is a good practice to split the dataset into training, test, and validation.
  4. `Dataset size:` the larger the dataset, the higher the generalization of the model to unseen data.
  5. `Samples per parameter in Neural Networks:` ideally, there should be 10x the number of samples (feature-label pairs) than parameters (weights) in a neural network, i.e, ten examples per weight.
  6. `Network depth:` the deeper the network (number of layers), the more information is extracted (learned).
  7. `Bias-variance trade-off:` while under-parametrization can cause bias (underfitting), over-parametrization can cause variance (overfitting). In mainstream machine learning, one tries to find a balance.
  8. `Double descent:` over-parametrization beyond certain interpolation threshold can lead to model generalization (good test performance). Moreover, as the size of the parameter vector to be optimized increases (as large as one million), the local minima get closer to the global minimum.
  9. `Batch size:` a larger batch size (increase memory resource) leads to a speed up in training and to a lower asymptotic test accuracy (classification), and hence a lower generalization to unseen data.
  10. `Learning rate:` a higher learning rate can speed-up training, however, too large a learning rate can make the loss function value to jiggle around the loss landscape and to never converge.

- **On the quantum side (NISQ devices):**
  1. `Coherence friendly:` the quantum circuit must be shallow, i.e, it must have a small number of gate layers executed in parallel (small [circuit depth](https://qiskit.org/documentation/_images/depth.gif)) in order to be executed during a time window shorter than the decoherence time.
  2. `Hardware friendly (qubit routing):` gate coupling must be allowed only between nearest-neighbor physical qubits etched into the hardware processor in order to avoid non-trivial transpilation, i.e, the use of SWAP gates (a non-native gate in most hardware platforms) during qubit routing (mapping from the circuit diagram to a hardware topology).
  3. `Small number of hyperparameters:` the algorithm must seek the minimum number of angles to be optimized in order to avoid classical optimization overhead (when classical computation becomes too expensive).
  
## Algorithm outline

1. Define a circuit ansatz and a list of available gates.

2. Define the cost function for classification with the quantum circuit. 

3. Perform the state preparation ([quantum embedding](https://pennylane.ai/qml/glossary/quantum_embedding.html)).

4. Initialize the trainable parameter values ($\theta_t$) of the fiducial state to random values.

5. Evaluate the quantum loss function $\mathcal{L}(\theta_t)$ in the quantum device.

6. Use the parameter shift rule to compute the gradient of the cost function with respect to its tunable parameters (gate angles).

7. Identify the operators (gates) with the maximum and minimum gradient in magnitude. Add to the ansatz the gate whose gradient is at maximum magnitude. Remove from the set the gate with the gradient at minimum magnitude if it is smaller than a predefined threshold. Here, we add one gate per iteration while deleting a single gate if it satisfies the above condition.

8. Feed the optimized $\theta_{t+1}$ parameter vector to the input of the quantum circuit.

9. Repeat step 6 onwards for $n$ epochs/iterations until convergence.

 
<!-- Installation: -->
# Installation
  
1. Clone this repository and access the cloned directory:
```bash
git clone https://github.com/QuCAI-Lab/adapt-qka.git && cd adapt-qka
```
2. Create a conda environment named "adapt-qka" and activate it:
```bash
conda create -yn adapt-qka python==3.8.8 && conda activate adapt-qka
```
3. Install pip in the current environment with conda and check for updates with pip:
```bash
conda install -yc conda-forge pip==23.0.1 && python -m pip install --user --upgrade pip
```
4. Install the core dependencies with the [requirements.txt](requirements.txt) file:
```bash
python -m pip install -r requirements.txt
```
5. Install graphviz with conda for the transpiler function:
```bash
conda install python-graphviz
```

The `python -m pip install .` command is equivalent to the `python -m setup.py install` command.

- Flags: 
  - The -m flag in `python -m pip` enforce the pip version tied to the active environment, i.e, instructs python to run pip as the __main__ module (script).
  - The [`--user`](https://pip.pypa.io/en/stable/cli/pip_install/#install-user) flag in `pip install` sets the installation path to the user install directory.
  - The `--no-deps` flag ensures that `setup.py` will not overwrite the conda dependencies that you have already installed using the `environment.yml` file. In this case, the pip-equivalent packages specified in the `requirements.txt` file will not be used.
  - The `-e` flag stands for editable mode (recommended for developers). It installs the package without copying any files to the interpreter directory allowing for source code changes to be instantly propagated to the code library without the need of rebuild and reinstall, however, the python process/kernel will need to be restarted. It sets the pacakge info: Build dev_0 and Channel \<develop>. It also creates the `ibm2022-open-sol.egg-info` file that enables the user to access the package information by: `conda list ibm2022-open-sol`. For more information, see the setuptools [development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html).
  - The `-v` flag enables progress display (verbose).
  
<!-- Quickstart: -->
# Quickstart

- Train and test the model:
```bash
python -m adapt_qka._main.qka
```

- Alternatively, run the package:
```python
# Import the package:
import adapt_qka as qka
# Display info:
qka.about()

'''Data preprocessing:'''
dataset=qka.preprocessing('iris.txt')
x_train, y_train = dataset['x_train'], dataset['y_train']
x_test, y_test = dataset['x_test'], dataset['y_test']
#print(f'\n{(x_train[:1][0] == x_train[0]).all()}') # >>> True

'''Simulator:'''
kernel = qka.AdaptQKA(data=dataset) # Using built-in parameters and gates.
# Show Kernel value between the first two data points:
print(kernel.kernel_value(x_train[0], x_train[1]))
# Show Kernel matrix between equal samples:
print(kernel.kernel_matrix(x_train[:1], x_train[:1]))

# Training parameters with circuit transpilation using default coupling map:
new_params, new_gates = kernel.train(epochs=1, threshold=1.0e-5)
# Show current quantum circuit:
kernel.show_kernel(x_train[0], x_train[0], new_params, new_gates, message='Current circuit with optimized parameters:')
```
You can also pass a real device name for custom transpilation:
```python
# Training parameters with circuit transpilation using custom coupling map:
provider=qka.load_ibm()
qubit_layout = get_qubit_layout('ibmq_manila', provider)
new_params, new_gates = kernel.train(epochs=1, threshold=1.0e-5, coupling_map=qubit_layout)
# Show current quantum circuit:
kernel.show_kernel(x_train[0], x_train[0], new_params, new_gates, message='Current circuit with optimized parameters:')
```
Finally, train the SVM:
```python
# Train the SVM:
svm = kernel.train_svm(new_params)

# Prediction with one sample:
kernel.prediction(svm, x_test[0].reshape(1, -1), y_test[0].reshape(1, -1))
# Show accuracy for the training dataset with the optimized parameters:
kernel.accuracy(svm, x_train, y_train)
# Show accuracy for the test dataset with the optimized parameters:
kernel.accuracy(svm, x_test, y_test)
```
For experiments with IBM hardware:
```python
'''Real device:'''
# Define the kernel for the real quantum device:
kernel = AdaptQKA(dataset, real_device='ibmq_lima')
# Training parameters:
params_device, new_gates = kernel.train(epochs=1, threshold=1.0e-5)
# Show current quantum circuit:
kernel.show_kernel(x_train[0], x_train[0], params_device, new_gates, message='Current circuit with optimized parameters:')
# Train the SVM:
svm = kernel.train_svm(params_device)
# Show accuracy for the whole training dataset with the optimized parameters:
print('\nAccuracy on training dataset:')
kernel.accuracy(svm, x_train, y_train)
```
 
# Work in progress

Create the RNN model (add support for PyTorch and TensorFlow) for parameter initialization:
```python
NUM_ANGLES=8
nn=qka.NeuralNetworkTorch(4)
rnn1=qka.RnnTf(NUM_ANGLES) # TensorFlow.
rnn2=qka.RnnTorch(NUM_ANGLES) # Torch.
nn # To visualize Torch model architecture.
```
Train the neural network-based QKA model to get the parameter initialization:
```python
# Under development...
initialization = qka.Train(data).neural_network_torch(neural_network=nn, epochs=10, cost=None)
```
  

<!-- Acknowledgement: -->
# Acknowledgement

The authors would like to acknowledge [Xanadu.ai](https://www.xanadu.ai/) and the [Open Hackathon Sponsors](https://github.com/XanaduAI/QHack2023#hackathon-prizes) for the opportunity.


<!-- References: -->
# References &nbsp; <a href="#"><img valign="middle" height="45px" src="https://img.icons8.com/book" width="45" hspace="0px" vspace="0px"></a> 

<a name="1"></a> \[1] J. R. Glick, T. P. Gujarati, A. D. Córcoles, Y. Kim, A. Kandala, J. M. Gambetta, K. Temme. Covariant quantum kernels for data with group structure [arXiv:2105.03406 (2021)](https://arxiv.org/abs/2105.03406).

<a name="2"></a> \[2] Harper R. Grimsley, Sophia E. Economou, Edwin Barnes, Nicholas J. Mayhall, "An adaptive variational algorithm for exact molecular simulations on a quantum computer". [Nat. Commun. 2019, 10, 3007](https://www.nature.com/articles/s41467-019-10988-2).

<a name="3"></a> \[3] J. Preskill, "Quantum computing in the NISQ era and beyond," [Quantum 2, 79 (2018)](https://quantum-journal.org/papers/q-2018-08-06-79/).

<a name="4"></a> \[4] Peter W Shor, "Algorithms for quantum computation: discrete logarithms and factoring," in [Proceedings 35th annual symposium on foundations of computer science](https://ieeexplore.ieee.org/document/365700) (Ieee, 1994) pp. 124–134.

<a name="5"></a> \[5] P. W. Shor. Polynomial-time algorithms for prime factorization and discrete logarithms on a quantum computer. [SIAM J. Comp.](https://doi.org/10.1137/S0097539795293172), 26(5):1484–1509, 1997.

<a name="6"></a> \[6] Lov K. Grover. A fast quantum mechanical algorithm for database search. In
[Proceedings of the Twenty-Eighth Annual ACM Symposium on Theory of Computing](https://dl.acm.org/doi/10.1145/237814.237866), STOC ’96, page 212–219, New York, NY, USA, 1996. Association for Computing Machinery.

<a name="7"></a> \[7] A. Peruzzo, J. McClean, P. Shadbolt, M.-H. Yung, X.-Q. Zhou, P. J. Love, A. Aspuru-Guzik, and J. L. O’Brien, "A variational eigenvalue solver on a photonic quantum processor", [Nature Communications 5, 4213 (2014)](https://www.nature.com/articles/ncomms5213).

<a name="8"></a> \[8] Nakanishi, K. M., Mitarai, K. & Fujii, K. Subspace-search variational quantum eigensolver for excited states. [Phys. Rev. Res. 1, 033062 (2019)](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.1.033062).

<a name="9"></a> \[9] Y. Liu, S. Arunachalam, and K. Temme, A rigorous and robust quantum speed-up in supervised machine learning. [arXiv:2010.02174](https://arxiv.org/abs/2010.02174) (2020).

<a name="10"></a> \[10] Thomas Hubregtsen, David Wierichs, Elies Gil-Fuster, Peter-Jan H. S. Derks, Paul K. Faehrmann, and Johannes Jakob Meyer, Training Quantum Embedding Kernels on Near-Term Quantum Computers, [Phys. Rev. A 106, 042431](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.106.042431).

<!-- License: -->
# License

This work is licensed under a [Apache License 2.0](LICENSE.md) license.

<hr>
