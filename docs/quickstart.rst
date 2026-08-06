.. _quickstart:

Quickstart
==========

This page walks through a complete optimization in about forty lines: preparing
the excited state of a qubit by tuning the amplitude and frequency of a drive.


If you have not installed ParaQeet yet, see :ref:`installation <install>` — in
short, ``pip install paraqeet``.

Define the control signal
-------------------------

In ParaQeet, the signal stack is designed to model the signal stack in experimental setups.
Every pulse is built from :class:`~paraqeet.signal.signal.Signal` components.
In this example, a constant envelope is mixed with a local oscillator by an IQ mixer. 
All tunable values are :class:`~paraqeet.quantity.Quantity` objects: bounded,
unit-aware parameters that any optimizer can adjust:

.. code-block:: python

   import numpy as np
   from paraqeet import (
       Expm, GOAT, OptimizationMap, Quantity, SchroedingerEquation,
       ScipyOptimizer, StateTransferFidelity,
   )
   from paraqeet.measurement.utils import overlap_state_vector
   from paraqeet.hamiltonian import Drive, QubitHamiltonian
   from paraqeet.signal import ConstantEnvelope, IQMixer

   freq = 4.8e9 * 2 * np.pi
   t_final = 10e-9

   generator = IQMixer(envelopes=[ConstantEnvelope()])
   amplitude, _, lo_freq, _ = generator.get_parameters()
   amplitude.set_value(0.8 * np.pi / t_final)
   lo_freq.set_value(1.01 * freq)

Model the system
----------------

The signal drives a qubit through a Pauli-X coupling, and the closed-system
dynamics is given by the Schrödinger equation:

.. code-block:: python

   qubit = QubitHamiltonian(frequency=Quantity(freq, 0.8 * freq, 1.2 * freq))
   qubit.drives = [Drive(np.array([[0.0, 1.0], [1.0, 0.0]]), generator)]
   model = SchroedingerEquation(
       hamiltonian_func=qubit.get_value,
       hamiltonian_gradient_func=qubit.get_gradient,
   )

Propagate and define the goal
-----------------------------

A propagation solves the equation of motion, and a fidelity measure turns the
final state into a scalar goal function. Wrapping the propagation in
:class:`~paraqeet.propagation.goat.GOAT` adds analytic gradients (using the GOAT method
:cite:p:`machnes2018tunable`), so the optimizer receives exact derivatives:

.. code-block:: python

   prop = GOAT(
       Expm(
           eom_func=model.get_value,
           resolution=100e9,
           initial_state=np.array([[1.0], [0.0]]),
       ),
       eom_gradient_func=model.get_gradient,
   )
   fidelity = StateTransferFidelity(
       propagation_func=prop.get_value,
       propagation_gradient_func=prop.get_gradient,
       target_state=np.array([[0.0], [1.0]]),
       overlap=overlap_state_vector,
   )

Optimize
--------

Collect the parameters to tune in an
:class:`~paraqeet.optimization_map.OptimizationMap` and hand everything to an
optimizer:

.. code-block:: python

   optmap = OptimizationMap()
   optmap.add(generator, [amplitude, lo_freq])
   opt = ScipyOptimizer(
       measure_func=fidelity.calculate_normalized_scalar,
       optimization_map=optmap,
   )
   result = opt.optimize(times=t_final)

``result.value`` is the final infidelity, and the optimized values are already
written back into the :class:`~paraqeet.quantity.Quantity` objects by the :class:`~paraqeet.optimization_map.OptimizationMap`;
print ``amplitude`` or ``lo_freq`` to see them.

Where to go next
----------------

- :doc:`concepts` — read about the layered architecture and how to choose an optimization
  method.
- :doc:`notebooks/01A_OptimizationMap` — how :class:`~paraqeet.quantity.Quantity` and
  :class:`~paraqeet.optimization_map.OptimizationMap` connect a model to an optimizer.
- :doc:`notebooks/01B_Gradient_evaluation` — how gradients flow through the
  package.
- The full :doc:`example gallery <notebooks/index>` and the API reference.
