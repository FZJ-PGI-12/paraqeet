Examples
========

We introduce the usage of the package with some examples. 
New users are recommended to start with :doc:`01A_OptimizationMap`, :doc:`01B_Gradient_evaluation` and
:doc:`02A_Single_qubit_state_preparation`.

Fundamentals
------------

- :doc:`01_OptimizationMap` — How parameters (``Quantity``) and the ``OptimizationMap`` connect a model to an optimizer.
- :doc:`01B_Gradient_evaluation` - A walkthrough over gradient computation across the package.

Single-qubit control
--------------------

- :doc:`02A_Single_qubit_state_preparation` — State preparation of a single spin by tuning a cosine drive.
- :doc:`02B_Single_qubit_gate` — Gradient-descent optimization of a single-qubit gate.
- :doc:`02C_Qubit-bayesian-optimization` — Gradient-free Bayesian optimization of a single-qubit gate.
- :doc:`03_DRAG_pulses` — DRAG correction of a Gaussian pulse to suppress leakage.

GRAPE
-----

- :doc:`04A_GRAPE_TLS` — GRAPE on a two-level system.
- :doc:`04B_Single_qubit_gate_GRAPE` — Single-qubit gate optimization with GRAPE.

GOAToverGRAPE for gradient based dCRAB
--------------------------------------

- :doc:`02D_GOAToverGRAPE_TLS` — Optimal control of a single spin using GOAT over GRAPE.
- :doc:`02E_GOAToverGRAPE_dCRAB` — Gradient-based dCRAB optimization of a single spin.
- :doc:`04C_Single_qubit_gate_GOAToverGRAPE` — Single-qubit gate optimization using GOAT over GRAPE.

Two-qubit and open systems
--------------------------

- :doc:`05_Two-qubit-cross-resonance` — Gradient-based optimization of a cross-resonance gate between two transmons.
- :doc:`06_Resonator_decay` — Decay of a coherent state of a resonator (open-system dynamics).

Building custom models
----------------------

- :doc:`07A_Custom_Hamiltonian` — Plug a user-defined Hamiltonian function into ParaQeet.
- :doc:`07B_Modelling_using_QuTiP` — Build the Hamiltonian using function and plug it into ParaQeet for optimization.

Bosonic systems and smoothness
------------------------------

- :doc:`08A_Smoothness_measure` — Constrain piece-wise-constant pulses to vary smoothly.
- :doc:`08B_Bosonic_grape_state_preparation` — Arbitrary bosonic state preparation using GRAPE.
- :doc:`08C_Bosonic_grape_with_smooth_pulses` — Bosonic state preparation with smooth pulses via gradient-based dCRAB / GOAT over GRAPE.

.. toctree::
   :maxdepth: 1
   :hidden:

   01A_OptimizationMap
   01B_Gradient_evaluation
   02A_Single_qubit_state_preparation
   02B_Single_qubit_gate
   02C_Qubit-bayesian-optimization
   02D_GOAToverGRAPE_TLS
   02E_GOAToverGRAPE_dCRAB
   03_DRAG_pulses
   04A_GRAPE_TLS
   04B_Single_qubit_gate_GRAPE
   04C_Single_qubit_gate_GOAToverGRAPE
   05_Two-qubit-cross-resonance
   06_Resonator_decay
   07A_Custom_Hamiltonian
   07B_Modeling_using_QuTiP
   08A_Smoothness_measure
   08B_Bosonic_grape_state_preparation
   08C_Bosonic_grape_with_smooth_pulses