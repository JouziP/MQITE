This repository contains modules and codes that are developed based on MQITE algorithm introduced in https://arxiv.org/abs/2208.10535v1 

The structure of this package is the following. 
The program has the following main packages:

* Amplitude: contains the class Amplitude that runs a circuit UQU|0> on a quasm simulators, computes the amplitudes |c_j|

* Phase: contains the class Phase which uses and Amplitude object to construct additional circuit (Appendix B) and computes the real and imaginary parts of every observed c_j from Amplitude. It also translate c_j into parameter (y_j).

* UpdateCircuit: uses the ccomputed c_j (real, imaginary) to update circuit. This is done by creating a layer of gates and adding them to the start of the current instruction.

* BasicFunctions: contains some trivial and toolkit functions that are used in the body of the main classes

* MultiQubitGate: is the core class/functions that translate e^{\sum_j y_j P_j} into one and two-qqubit gates. See Appendix A for detail.

* Benchmarks: contains classes that use the amplitude and phase instances to test and measure certain metrics, or benchmark against state-vector

* Problems: example problems that can be used to tets the algorithm

* Observables: A class that uses the current state of the circuit to calculate the expectaion ogf an observable. Current implementation uses state-vector.

The main entry is SimulatMQITE. This folder contains one example implementation of MQITE Algorithm (fig.1 in https://arxiv.org/abs/2208.10535v1)
For start, you may want to run the SimulatMQITE/Tests/ising_model.py which is a test application.




