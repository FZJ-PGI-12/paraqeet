.. _quickstart:

Quickstart
==========

This page walks through a complete optimization in about forty lines: preparing
the excited state of a qubit by tuning the amplitude and frequency of a drive.
It is the same example that ships (and is tested) as the package docstring of
:mod:`paraqeet`.

If you have not installed ParaQeet yet, see :ref:`installation <install>` — in
short, ``pip install paraqeet``.

Define the control signal
-------------------------

Every pulse is built from :class:`~paraqeet.signal.waveform.Waveform` components.
Here a constant envelope is mixed with a local oscillator by an IQ mixer. All
tunable values are :class:`~paraqeet.quantity.Quantity` objects — bounded,
unit-aware parameters that any optimizer can adjust:

.. code-block:: python

   import numpy as np
   from paraqeet import (
       OptimizationMap, Quantity, SchroedingerEquation,
       ScipyExpmGOAT, ScipyOptimizer, StateTransferFidelity,
   )
   from paraqeet.measurement.utils import overlap_state_vector
   from paraqeet.model.drive import Drive
   from paraqeet.model.qubit import QubitHamiltonian
   from paraqeet.signal.envelopes import ConstantEnvelope
   from paraqeet.signal.iq_mixer import IQMixer

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

A propagator solves the equation of motion, and a fidelity measure turns the
final state into a scalar goal function. ``ScipyExpmGOAT`` also propagates the
analytic gradients (GOAT), so the optimizer receives exact derivatives:

.. code-block:: python

   prop = ScipyExpmGOAT(
       eom_func=model.get_value, eom_gradient_func=model.get_gradient,
       resolution=100e9, initial_state=np.array([[1.0], [0.0]]),
   )
   fidelity = StateTransferFidelity(
       propagation_func=prop.propagate,
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
written back into the ``Quantity`` objects — print ``amplitude`` or ``lo_freq``
to see them.

Where to go next
----------------

- :doc:`notebooks/01A_OptimizationMap` — how ``Quantity`` and
  ``OptimizationMap`` connect a model to an optimizer.
- :doc:`notebooks/01B_Gradient_evaluation` — how gradients flow through the
  package.
- :doc:`concepts` — the layered architecture and how to choose an optimization
  method.
- The full :doc:`example gallery <notebooks/index>` and the API reference.
