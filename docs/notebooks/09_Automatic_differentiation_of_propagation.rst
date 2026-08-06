Computing gradients of propagation by Automatic differentiation
===============================================================

In this example, we demonstrate stepwise how gradient of propagation
module can be computed using automatic differentation. For a minimal
working example, here we unpack the ``Expm`` method and compute the
gradients by AD. Finally, we compare the computed gradients with GOAT
for verification.

Since we use a modular architecture for the package, the propagation
method computes the equation of motion (EOM) for some given times
(specified by the initial and final time, and the resolution). Thus the
gradient in this case has to be computed using the chain rule

.. math:: \frac{\partial \ket{\psi(t)}}{\partial \alpha} = \sum_{\tau \in [0, t]} \frac{\partial \ket{\psi(t)}}{\partial H(\tau)} \frac{\partial H(\tau)}{\partial \alpha} 

where :math:`\alpha` is some pulse parameter. The first term on the
right can be computed by automatic differentiation of the propagation
method, and the second term is obtained through the derivative of the
Hamiltonian (or the EOM).

In case, the propagation is performed in a loop (by checkpointing), an
additional term is added to the above equation

.. math:: \frac{\partial \ket{\psi(t)}}{\partial \alpha} = \sum_{\tau \in [0, t]} \frac{\partial \ket{\psi(t)}}{\partial H(\tau)} \frac{\partial H(\tau)}{\partial \alpha} + \sum_{\tau \in [0, t_1, t_2, ...]} \frac{\partial \ket{\psi(t)}}{\partial \ket{\psi(\tau)}} \frac{\partial \ket{\psi(\tau)}}{\partial \alpha}.

We only consider the former in this example, and advise the reader to
take a look at the class ``AutoDiffGradients`` for the latter.

Let’s define a simple system of a driven qubit.

.. code:: ipython3

    import numpy as np
    
    from paraqeet.eom.schroedinger_equation import SchroedingerEquation
    from paraqeet.hamiltonian.drive import Drive
    from paraqeet.hamiltonian.qubit import QubitHamiltonian
    from paraqeet.optimization_map import OptimizationMap
    from paraqeet.quantity import Quantity
    from paraqeet.signal.envelopes import ConstantEnvelope
    from paraqeet.signal.iq_mixer import IQMixer

.. code:: ipython3

    freq_q = 4.8e9
    omega_q = 2 * np.pi * freq_q
    
    envelope = ConstantEnvelope()
    gen = IQMixer(envelopes=[envelope])
    
    qubit_hamiltonian = QubitHamiltonian(frequency=Quantity(omega_q, 0.8 * omega_q, 1.2 * omega_q, unit="Hz"), drives=[])
    drive = Drive(qubit_hamiltonian.sigma_x, gen)
    qubit_hamiltonian.drives = [drive]
    
    schrgl = SchroedingerEquation(
        hamiltonian_func=qubit_hamiltonian.get_value,
        hamiltonian_gradient_func=qubit_hamiltonian.get_gradient,
    )
    
    params_tone = envelope.get_parameters()
    params_gen = gen.get_parameters()
    params_gen




.. parsed-literal::

    [Amplitude: 24.7 MHz x 2pi,
     t_final: 32 ns,
     lo_freq: 4.8 GHz x 2pi,
     Phase: 0 rad]



Add parameters to the optmap and register the optimizable parameters

.. code:: ipython3

    t_simu = 10e-9
    params_gen[0].set_value(0.8 * np.pi / t_simu)
    params_gen[2].set_value(1.01 * omega_q)
    
    optmap = OptimizationMap()
    optmap.add(gen, [params_gen[0], params_gen[1], params_gen[2]])
    optmap.register_params_with_optimizables()

Here we unpack the inner workings of ``Expm`` to construct an easy
example to work with. Lets first compute the EOM for the interpolated
times (given a resolution), which we will then pass to the propagate
method

.. code:: ipython3

    import jax.numpy as jnp
    
    from paraqeet.propagation.utils import construct_times
    
    init = np.array([[1.0], [0.0j]])  # |0>
    target = np.array([[0.0j], [1.0]])  # |1>
    
    time = jnp.array([0.0, t_simu])
    resolution = 100e9
    eom_func = schrgl.get_value
    
    time_grid = []
    eoms = []
    
    for ti in range(1, len(time)):
        times, dt = construct_times(time, ti, resolution)
        eom = eom_func(times + dt / 2) * dt
        time_grid.append(times + dt / 2)
        eoms.append(eom)
    
    time_grid = jnp.hstack(time_grid)
    eoms = jnp.hstack(eoms)
    steps_arr = jnp.arange(0, len(time_grid), 1)

The following are the main functions in ``Expm`` that propagate a state
by exponentiating the EOM:

.. math:: \ket{\psi(t)} = e^{-iH(t- t')}\ket{\psi(t')}.

The ``propagate`` method here performs the state evolution and takes the
EOM as an input

.. code:: ipython3

    from typing import Any
    
    from jax import jit
    from jax.lax import scan
    from jax.scipy.linalg import expm
    
    from paraqeet.quantity import Array
    
    
    @jit
    def propagate(eom: Array, psis_t: Array, steps_arr: Array) -> Array:
        """Propagate the system in time.
    
        Iteratively propagate state/states (psis_t) according
        to the equation of motion (eom). The eom is exponentiated using
        ``jax.scipy.linalg.expm`` to compute the propagators.
        The iterations use ``jax.lax.scan`` to avoid compilation overhead.
    
        Args:
            psis_t: State/states at time 't'.
            eom: Equation of motion for a list of times.
            steps_arr: Array from 0 to the length of the list of times, in steps
                of 1 representing the iteration index.
    
        Returns:
            The evolved state.
    
        """
    
        def propagate_body(psis_t: Array, index: Any) -> tuple[Array, Array]:
            psis_t = propagate_psi(eom[index], psis_t)
            return psis_t, psis_t
    
        psis_t, _ = scan(propagate_body, psis_t, steps_arr)
    
        return psis_t
    
    
    @jit
    def propagate_psi(eom_matrix: Array, psis_t: Array) -> Array:
        """Propagate the state/states (psis_t).
    
        Args:
            eom_matrix: The equations of motion matrix.
            psis_t: State/states at time 't'.
    
        Returns:
            Array: Returns the evolved state.
    
        """
        propagated: Array = expm(eom_matrix) @ psis_t
        return propagated

.. code:: ipython3

    propagate(eoms, jnp.array(init), steps_arr)




.. parsed-literal::

    Array([[ 0.68933469-0.42125643j],
           [-0.58818818-0.03735471j]], dtype=complex128)



Let’s compute the gradient of the evolved state with respect to the
Hamiltonian. Here we use the ``get_value_and_jacobian_rev`` function
defined in autograd_utils

.. code:: ipython3

    from paraqeet.autograd_utils import get_value_and_jacobian_rev
    
    prop_grad_func = get_value_and_jacobian_rev(propagate, argnums=0)
    val, grads = prop_grad_func(eoms, jnp.array(init), steps_arr)

.. code:: ipython3

    grads.shape




.. parsed-literal::

    (2, 1, 1000, 2, 2)



We need to sum over the gradient of the EOM at different time

.. code:: ipython3

    _, schrgl_grads = schrgl.get_value_and_gradient(time_grid)

.. code:: ipython3

    schrgl_grads.shape




.. parsed-literal::

    (1000, 3, 2, 2)



Here the second index is for the number of parameters.

.. code:: ipython3

    grads1 = jnp.transpose(grads, axes=(2, 0, 1, 3, 4))
    grads1.shape




.. parsed-literal::

    (1000, 2, 1, 2, 2)



We need to sum over times and the last two indices. Using the notation
“t -> times”, “n,m-> dim of state vector”, “p -> no. of parameters”, and
“j, k -> dim of Hamiltonian”, we perform the following summation

.. code:: ipython3

    final_grads = jnp.einsum("tnmjk, tpjk -> pnm", grads1, schrgl_grads) * dt
    final_grads.shape




.. parsed-literal::

    (3, 2, 1)



Comparison with GOAT
--------------------

Let’s compute the same gradient using GOAT and compare the gradient
values

.. code:: ipython3

    freq_q = 4.8e9
    omega_q = 2 * np.pi * freq_q
    
    envelope = ConstantEnvelope()
    gen = IQMixer(envelopes=[envelope])
    
    qubit_hamiltonian = QubitHamiltonian(frequency=Quantity(omega_q, 0.8 * omega_q, 1.2 * omega_q, unit="Hz"), drives=[])
    drive = Drive(qubit_hamiltonian.sigma_x, gen)
    qubit_hamiltonian.drives = [drive]
    
    schrgl = SchroedingerEquation(
        hamiltonian_func=qubit_hamiltonian.get_value,
        hamiltonian_gradient_func=qubit_hamiltonian.get_gradient,
    )
    
    params_tone = envelope.get_parameters()
    params_gen = gen.get_parameters()
    params_gen




.. parsed-literal::

    [Amplitude: 24.7 MHz x 2pi,
     t_final: 32 ns,
     lo_freq: 4.8 GHz x 2pi,
     Phase: 0 rad]



.. code:: ipython3

    t_simu = 10e-9
    params_gen[0].set_value(0.8 * np.pi / t_simu)
    params_gen[2].set_value(1.01 * omega_q)

.. code:: ipython3

    from paraqeet.propagation import GOAT, Expm
    
    init = np.array([[1.0], [0.0]])  # |0>
    target = np.array([[0.0], [1.0]])  # |1>
    
    times = np.array([0.0, t_simu])
    
    prop = GOAT(
        Expm(eom_func=schrgl.get_value, resolution=100e9, initial_state=init),
        eom_gradient_func=schrgl.get_gradient,
    )  # implicit timestep is 1 / resolution

.. code:: ipython3

    optmap = OptimizationMap()
    optmap.add(gen, [params_gen[0], params_gen[1], params_gen[2]])
    optmap.register_params_with_optimizables()

.. code:: ipython3

    val, grads = prop.get_value_and_gradient(np.array([0.0, t_simu]))

.. code:: ipython3

    jnp.allclose(grads[-1], final_grads)




.. parsed-literal::

    Array(True, dtype=bool)



We can see that the gradients from AD match with the gradient computed
using GOAT.
