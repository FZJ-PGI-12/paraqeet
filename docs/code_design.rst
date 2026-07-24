.. _code_design:

Code design: extending ParaQeet
===============================

ParaQeet is built to be extended by subclassing. Every layer of the package —
``signal``, ``hamiltonian``, ``eom``, ``propagation``, ``measurement``, ``optimizer``
— is defined by a small *abstract base class*. 
To plug in your own pulse shape, Hamiltonian, solver, goal function, or optimizer,
subclass the corresponding base class and
implement its handful of abstract methods; the new component then drops into the
existing pipeline without changes anywhere else.


The two root interfaces
-----------------------

There are two fundamental base classes in ParaQeet, the ``Differentiable`` 
and the ``Optimizable`` base class. 

:class:`~paraqeet.differentiable.Differentiable`
   Anything that can be differentiated to obtain *its gradient* with respect to its
   optimizable parameters. Subclasses implement:

   - ``get_value(times) -> Array | Float``
   - ``get_gradient(times) -> Array``

   A concrete ``get_value_and_gradient(times)`` is provided and simply calls the
   two. Override it when value and gradient are cheaper to compute together, 
   for e.g., while using automatic differentiation, to reduce redundant computation.

:class:`~paraqeet.optimizable.Optimizable`
   Anything that exposes tunable :class:`~paraqeet.quantity.Quantity` parameters
   to the optimizer. Subclasses implement a single method:

   - ``get_parameters() -> list[Quantity]``

   Everything else — ``name``, ``optimizable_parameters``,
   ``set_optimizable_parameters()``, ``_is_optimized()`` — is provided by the
   base class. The optimizer collects these parameters *by reference* and writes
   optimized values straight back into them.

Most template classes below inherit from **both** (``Optimizable`` for their
parameters, ``Differentiable`` for their value/gradient).


A note on the gradient computation, 

- **Signal.** :class:`~paraqeet.signal.waveform.Waveform` automates gradient 
  evaluation: implement the ``_evaluate`` method as a *pure JAX function* (arguments
  ordered as ``(parameters…, t)``, returning a scalar for scalar ``t``) and the base
  class derives ``get_value`` and the parameter gradients automatically through
  JAX autodiff. Note that ordering of the parameters as an input to ``_evaluate`` 
  should **always** be the same as the ``get_parameters`` method.

- A similar fallback to automatic differentiation for the ``model`` and ``propagation``
  module is a work in progress (refer to branch `99-autodiff-as-fallback`_ 
  as a reference), and would soon be added in a future release.

- The module ``autodiff_utils`` provides easy to use wrappers around JAX vjp and jvp
  methods that can be used by the user to construct their own ``get_value_and_gradient`` methods.

- Refer to :doc:`notebooks/01B_Gradient_evaluation` for how gradients are computed 
  in each module of the package.


Finally, we recommend the reader to refer to Example :doc:`notebooks/07A_Custom_Hamiltonian` 
for creating their own models by inheriting from the provided base classes, and 
:doc:`notebooks/07B_Modeling_using_QuTiP` for using QuTiP :cite:p:`lambert2026qutip` based models with ParaQeet. 


.. _99-autodiff-as-fallback: https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/tree/99-autodiff-as-fallback