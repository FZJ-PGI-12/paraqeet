Using QuTiP with ParaQeet
=========================

In this example we demonstrate how QuTiP [Lambert2026] objects could be
used for modeling a quantum system, and combined with ParaQeet for
propagation and optimization tasks. Here we follow the same example as
before, of a state preparation task, but use QuTiP functions for
modeling the system.

.. code:: ipython3

    import jax.numpy as jnp
    import qutip as qt
    import qutip_jax
    
    import paraqeet as pq
    
    qutip_jax.set_as_default()

Here we use the `qutip-jax <https://github.com/qutip/qutip-jax>`__
package for better integration with ParaQeet, as ParaQeet deals with
pure JAX objects.

1. Define the parameters, Hamiltonian function and gradient functions
---------------------------------------------------------------------

Similar to the previous example, here we also define a ``CosEnvelope``
as our pulse ansatz.

The Hamiltonian in this example is, similarly, given by

.. math:: H(t) = \frac{\omega}{2} \sigma_z + u(A, \omega_d, t) \sigma_x, 

with :math:`u(A, \omega_d, t) = A\cos(\omega_d t)`.

.. code:: ipython3

    from paraqeet.signal.envelopes import Envelope
    
    amplitude = pq.Quantity(
        value=jnp.array(1.55e8),
        min_value=jnp.array(0.0),
        max_value=jnp.array(5 * 1e8),
        unit="Hz",
        name="Amplitude",
        two_pi=True,
    )
    
    frequency = pq.Quantity(
        value=jnp.array(4.8e9 * 2 * jnp.pi),
        min_value=jnp.array(0.8 * 4.8e9 * 2 * jnp.pi),
        max_value=jnp.array(1.2 * 4.8e9 * 2 * jnp.pi),
        unit="Hz",
        name="lo_freq",
        two_pi=True,
    )
    
    t_final = 12e-9
    t_final_qty = pq.Quantity(
        value=jnp.array(t_final),
        min_value=jnp.array(0.8 * t_final),
        max_value=jnp.array(1.2 * t_final),
        unit="s",
        name="t_final",
    )
    
    tls_freq = 4.8e9 * 2 * jnp.pi
    
    
    class CosEnvelope(Envelope):
        """Cosine envelope."""
    
        def __init__(self, amplitude, frequency, t_final=None):
            super().__init__(amplitude, t_final)
            self.frequency = frequency
    
        def get_parameters(self):
            """Return parameters amplitude and frequency"""
            return [self.amplitude, self.frequency]
    
        def _evaluate(self, amp, freq, t: pq.Array):
            """Define a cosine envelope."""
            return amp * jnp.cos(freq * t)
    
        def get_value(self, times):
            """Return value of the Envelope."""
            amp = self.amplitude.get_value()
            freq = self.frequency.get_value()
            return self._evaluate(amp, freq, times)
    
    
    cos_env = CosEnvelope(amplitude, frequency, t_final)
    
    
    # We define a H(t) for a single time-point t using QuTiP and pass it to the qt_func_to_jax_func wrapper
    # to convert into a JAX function which supports vectorized time-points as input.
    def tls_hamiltonian(t: float):
        """Define a Two level system Hamiltonian."""
        return 0.5 * tls_freq * qt.sigmaz() + qt.sigmax() * cos_env.get_value(t)

.. code:: ipython3

    tls_hamiltonian(0.0)




.. math::

    Quantum object: dims=[[2], [2]], shape=(2, 2), type='oper', dtype=JaxArray, isherm=True$$\left(\begin{array}{cc}1.508\times10^{ 10 } & 1.550\times10^{ 8 }\\1.550\times10^{ 8 } & -1.508\times10^{ 10 }\end{array}\right)



We can see that the output of the tls_hamiltonian is a ``Qobj``. For
working with ParaQeet we convert the Hamiltonian function to a jax
compatible function by using the wrapper function
``qt_func_to_jax_func``.

.. code:: ipython3

    from paraqeet.model.utils import qt_func_to_jax_func
    
    jax_ham_func = qt_func_to_jax_func(tls_hamiltonian)
    jax_ham_func(jnp.array([0.0, 1e-9]))




.. parsed-literal::

    Array([[[ 1.50796447e+10+0.j,  1.55000000e+08+0.j],
            [ 1.55000000e+08+0.j, -1.50796447e+10+0.j]],
    
           [[ 1.50796447e+10+0.j,  4.78976341e+07+0.j],
            [ 4.78976341e+07+0.j, -1.50796447e+10+0.j]]], dtype=complex128)



The ``jax_ham_func`` now works with a batch of time points and returns
the Hamiltonian as a JAX ``Array`` for each time point.

Similar to the previous example, for cases where the gradients of the
Hamiltonian are not required, for e.g., for state propagation and
gradient-free optimization the ``hamiltonian_and_gradient_func`` can be
defined by any function. Here we define a function that raises exception
when it is called.

.. code:: ipython3

    from paraqeet.model.schroedinger_equation import SchroedingerEquation
    
    
    def ham_grad(t):
        """Gradient of the Hamiltonian. Raises exception when called."""
        raise Exception("Hamiltonian gradients have not been defined.")
    
    
    model = SchroedingerEquation(hamiltonian_func=jax_ham_func, hamiltonian_and_gradient_func=ham_grad)

2. Define propagation method and measurement function
-----------------------------------------------------

Here we pick the standard ``ScipyExpmGOAT`` method for propagation and
``UnitaryFidelity`` as our measurement function. We start from the
identity matrix with the goal to prepare the Hadamard gate.

.. code:: ipython3

    from paraqeet.measurement.state_transfer_fidelity import StateTransferFidelity
    from paraqeet.measurement.utils import overlap_state_vector
    from paraqeet.model.utils import qobj_to_array
    from paraqeet.propagation.scipy_expm_goat import ScipyExpmGOAT
    
    init_qobj = qt.basis(2, 0)  # |0>
    target_qobj = qt.basis(2, 1)  # |1>
    
    # Convert Qobj to jax Array
    init = qobj_to_array(init_qobj)
    target = qobj_to_array(target_qobj)
    
    prop = ScipyExpmGOAT(
        eom_func=model.get_value, eom_and_grad_func=model.get_value_and_gradient, resolution=100e9, initial_state=init
    )
    times = jnp.array([0.0, t_final])
    
    zeroone = StateTransferFidelity(
        propagation_func=prop.propagate,
        propagation_and_gradient_func=prop.get_value_and_gradient,
        target_state=target,
        overlap=overlap_state_vector,
    )

.. code:: ipython3

    from plotting import plot_signal_and_dynamics
    
    ts = jnp.linspace(0.0, t_final, 501)
    plot_signal_and_dynamics(cos_env, prop, ts, state_labels=[r"$|0\rangle$", r"$|1\rangle$"]);



.. image:: 07B_Modeling_using_QuTiP_files/07B_Modeling_using_QuTiP_15_0.png


.. code:: ipython3

    zeroone.measure(times)




.. parsed-literal::

    Array(0.64040275, dtype=float64)



3. Gradient based optimization
------------------------------

For gradient-based optimization let’s define the gradient of the
Hamiltonian function also using QuTiP objects.

.. code:: ipython3

    def grad_tls_hamiltonian(t: pq.Array):
        """Value and gradient of the TLS Hamiltonian."""
        _, grads = cos_env.get_value_and_gradient(t)
        sigma_x = qobj_to_array(qt.sigmax())
        ham_grads = jnp.stack(
            [grads[:, 0].reshape((-1, 1, 1)) * sigma_x, grads[:, 1].reshape((-1, 1, 1)) * sigma_x], axis=1
        )
        return jax_ham_func(t), ham_grads

Finally we can redefine the model (and propagation and measurement
function) with the updated ``hamiltonian_and_gradient_func``.

.. code:: ipython3

    model = SchroedingerEquation(hamiltonian_func=jax_ham_func, hamiltonian_and_gradient_func=grad_tls_hamiltonian)
    
    prop = ScipyExpmGOAT(
        eom_func=model.get_value, eom_and_grad_func=model.get_value_and_gradient, resolution=100e9, initial_state=init
    )
    times = jnp.array([0.0, t_final])
    
    zeroone = StateTransferFidelity(
        propagation_func=prop.propagate,
        propagation_and_gradient_func=prop.get_value_and_gradient,
        target_state=target,
        overlap=overlap_state_vector,
    )

.. code:: ipython3

    zeroone.measure(times)




.. parsed-literal::

    Array(0.64040275, dtype=float64)



.. code:: ipython3

    cos_env.get_parameters()




.. parsed-literal::

    [Amplitude: 24.7 MHz x 2pi, lo_freq: 4.8 GHz x 2pi]



4. Create optmap and optimize
-----------------------------

.. code:: ipython3

    from paraqeet.optimization_map import OptimizationMap
    from paraqeet.optimizers.scipy_optimizer_gradient import ScipyOptimizerGradient
    
    optmap = OptimizationMap()
    optmap.add(cos_env)
    opt = ScipyOptimizerGradient(measure_and_gradient_func=zeroone.get_value_and_gradient, optimization_map=optmap)

.. code:: ipython3

    opt.optimize(times)


.. parsed-literal::

    Iteration    1 | Infid = 1.887855e-02
    Iteration    2 | Infid = 1.238241e-02


.. parsed-literal::

    Iteration    3 | Infid = 2.082279e-04
    Iteration    4 | Infid = 1.173600e-05
    Iteration    5 | Infid = 1.021922e-08


.. parsed-literal::

    Iteration    6 | Infid = 2.795519e-11
    Iteration    7 | Infid = 5.551115e-15




.. parsed-literal::

    {'status': 1, 'value': 5.551115123125783e-15, 'iterations': 10, 'message': 'CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL'}



.. code:: ipython3

    plot_signal_and_dynamics(cos_env, prop, ts, state_labels=[r"$|0\rangle$", r"$|1\rangle$"]);



.. image:: 07B_Modeling_using_QuTiP_files/07B_Modeling_using_QuTiP_26_0.png


.. code:: ipython3

    zeroone.measure(times)




.. parsed-literal::

    Array(1., dtype=float64)



References
----------

[Lambert2026] Lambert, Neill, et al. “QuTiP 5: The quantum toolbox in
Python.” Physics Reports 1153 (2026): 1-62.
