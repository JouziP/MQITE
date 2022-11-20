This repository contains modules and codes that are developed based on MQITE algorithm introduced in https://arxiv.org/abs/2208.10535v1 

The structure of this package is the following. 
The program has the following main packages:

* Amplitude: contains the class Amplitude that runs a circuit UQU|0> on a quasm simulators, computes the amplitudes |c_j|
* Phase: contains the class Phase which uses and Amplitude object to construct additional circuit (Appendix B) and computes the real and imaginary parts of every observed c_j from Amplitude. It also translate c_j into parameter (y_j).
* UpdateCircuit: uses the ccomputed c_j (real, imaginary) to update circuit. This is done by creating a layer of gates and adding them to the start of the current instruction.
* BasicFunctions: contains some trivial and toolkit functions that are used in the body of the main classes
* MultiQubitGate: is the core class/functions that translate e^{\sum_j y_j P_j} into one and two-qqubit gates. See Appendix A for detail.
* 

The main entry is SimulatMQITE. This folder contains one example implementation of MQITE Algorithm (fig.1 in https://arxiv.org/abs/2208.10535v1)



.
├── Amplitude
│   ├── Tests
│   │   ├── simulator_log.log
│   │   └── test_ampl.py
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── Amplitude.cpython-38.pyc
│   │   ├── __init__.cpython-38.pyc
│   │   ├── computeAmplitude.cpython-38.pyc
│   │   ├── computeAmplitudeV1.cpython-38.pyc
│   │   ├── computeAmplitudeV2.cpython-38.pyc
│   │   ├── getAmplitudes.cpython-38.pyc
│   │   ├── getBenchmark_after.cpython-38.pyc
│   │   ├── getBenchmark_before.cpython-38.pyc
│   │   └── getIndexsFromExecute.cpython-38.pyc
│   ├── amplitude.py
│   ├── computeAmplitudeV1.py
│   ├── computeAmplitudeV2.py
│   └── simulator_log.log
├── BasicFunctions
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-38.pyc
│   │   ├── exact_calculations.cpython-38.pyc
│   │   ├── functions.cpython-38.pyc
│   │   ├── getQCirc.cpython-38.pyc
│   │   ├── getRandomQ.cpython-38.pyc
│   │   ├── getRandomU.cpython-38.pyc
│   │   ├── getState.cpython-38.pyc
│   │   ├── getStateVectorValuesOfAmpl.cpython-38.pyc
│   │   ├── getUQUCirc.cpython-38.pyc
│   │   └── test_functions.cpython-38.pyc
│   ├── exact_calculations.py
│   ├── functions.py
│   ├── getQCirc.py
│   ├── getRandomQ.py
│   ├── getRandomU.py
│   ├── getState.py
│   ├── getStateVectorValuesOfAmpl.py
│   ├── getUQUCirc.py
│   ├── simulator_log.log
│   └── test_functions.py
├── Benchmarks
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-38.pyc
│   │   └── amplitudeBenchmark.cpython-38.pyc
│   └── amplitudeBenchmark.py
├── MultiQubitGate
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-38.pyc
│   │   └── functions.cpython-38.pyc
│   └── functions.py
├── Observables
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-38.pyc
│   │   └── observables.cpython-38.pyc
│   └── observables.py
├── Phase
│   ├── PhaseFunctions
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── computeAmplFromShots.cpython-38.pyc
│   │   │   ├── getImagPart_base_circ.cpython-38.pyc
│   │   │   ├── getImagPart_ref_circ.cpython-38.pyc
│   │   │   ├── getRealPart_base_circ.cpython-38.pyc
│   │   │   └── getRealPart_ref_circ.cpython-38.pyc
│   │   ├── computeAmplFromShots.py
│   │   ├── getImagPart.py
│   │   ├── getImagPart_base_circ.py
│   │   ├── getImagPart_ref_circ.py
│   │   ├── getRealPart.py
│   │   ├── getRealPart_base_circ.py
│   │   ├── getRealPart_ref_circ.py
│   │   └── simulator_log.log
│   ├── Tests
│   │   ├── __init__.py
│   │   ├── simulator_log.log
│   │   ├── test_phase.py
│   │   ├── test_phase_1.py
│   │   └── test_phase_getY.py
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-38.pyc
│   │   └── phase.cpython-38.pyc
│   ├── phase.py
│   └── simulator_log.log
├── Phase_
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── Phase.cpython-38.pyc
│   │   ├── __init__.cpython-38.pyc
│   │   ├── computePhaseStateVec_test.cpython-38.pyc
│   │   ├── getCosPhasesNoC0_test.cpython-38.pyc
│   │   └── getSinPhasesNoC0_test.cpython-38.pyc
│   ├── computePhaseStateVec_test.py
│   ├── getCosPhasesNoC0_test.py
│   └── getSinPhasesNoC0_test.py
├── Problems
│   ├── RydbergAtoms.py
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-38.pyc
│   │   ├── createPermutations.cpython-38.pyc
│   │   ├── getExactHamilt.cpython-38.pyc
│   │   ├── getNuclearExample.cpython-38.pyc
│   │   └── spinProblems.cpython-38.pyc
│   ├── createPermutations.py
│   ├── fermionOnLattice.py
│   ├── getExactHamilt.py
│   ├── getNuclearExample.py
│   ├── ham-0p-pn-spherical.txt
│   ├── ham-0p-spherical.txt
│   ├── ham-0p32.txt
│   ├── ham-JW-full.txt
│   ├── randomHamiltonian.py
│   └── spinProblems.py
├── README.md
├── Results
├── SimulateMQITE
│   ├── Simulator.py
│   ├── Tests
│   │   ├── ising_model.py
│   │   └── simulator_log.log
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── Simulator.cpython-38.pyc
│   │   ├── __init__.cpython-38.pyc
│   │   ├── functions.cpython-38.pyc
│   │   ├── functionsV2.cpython-38.pyc
│   │   ├── functionsV3.cpython-38.pyc
│   │   ├── log_config.cpython-38.pyc
│   │   ├── main2.cpython-38.pyc
│   │   ├── main_version1.cpython-38.pyc
│   │   └── main_version2.cpython-38.pyc
│   ├── log_config.py
│   ├── main_version1.py
│   ├── simulator_log.log
│   └── test_log.py
├── Tests
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-38.pyc
│   │   └── logger_4testing.cpython-38.pyc
│   └── logger_4testing.py
├── UpdateCircuit
│   ├── UpdateCircuit.py
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── UpdateCircuit.cpython-38.pyc
│   │   ├── __init__.cpython-38.pyc
│   │   ├── findCircuitParametersV1.cpython-38.pyc
│   │   ├── findComponents.cpython-38.pyc
│   │   ├── findComponentsV1.cpython-38.pyc
│   │   ├── findComponentsV2.cpython-38.pyc
│   │   └── updateCirc.cpython-38.pyc
│   ├── findCircParams.py
│   ├── findCircuitParametersV1.py
│   └── updateCirc.py
└── main_phaseValidation.py


