.. image:: ../big_logo.png
   :align: center
   :width: 90%
   :alt: logo
   :class: only-light

.. image:: _static/big_logo_dark.png
   :align: center
   :width: 90%
   :alt: logo
   :class: only-dark

===================================================================================
ParaQeet -  A quantum optimal control toolkit with simple parameter management
===================================================================================

Choose a pulse parametrization, simulate a quantum system, and optimize.

ParaQeet combines quantum optimal control methods with automatic differentiation
via JAX, aimed at resource efficient computation. Currently implemented
optimization methods:

- GRAPE :cite:p:`khaneja2005optimal`: Gradient Ascent Pulse Engineering
- GOAT :cite:p:`machnes2018tunable`: Gradient Optimization of Analytic conTrols
- dCRAB :cite:p:`rach2015dressing`: (Gradient based) dressed Chopped RAndom Basis
- GOAToverGRAPE: A variant of GROUP :cite:p:`sorensen2018quantum` that optimizes continuous pulse parameters with GRAPE inside.
- AD: Automatic differentiation of the state/propagator evolution

Currently implemented propagation methods:

- Expm: Matrix exponential using JAX ``expm`` :cite:p:`jax2018github`
- ExpmChebyshev: Matrix exponential using Chebyshev polynomial expansion :cite:p:`talezer1984accurate`
- ODE solvers: Diffrax :cite:p:`kidger2021on`, Verner 7th order method :cite:p:`verner2010numerically`, and Scipy Runge-Kutta methods :cite:p:`2020SciPy-NMeth`.

The propagation methods can be combined with the QOC methods leading to combinations such as GOAT QOC using ``ExpmChebyshev``.

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      Install ParaQeet from PyPI or set up a development environment.

   .. grid-item-card:: Quickstart
      :link: quickstart
      :link-type: doc

      Quickly setup of an optimization problem with ParaQeet.

   .. grid-item-card:: Examples
      :link: notebooks/index
      :link-type: doc

      A walkthrough of the capabilities of ParaQeet with physically motivated problems.

   .. grid-item-card:: Code design
      :link: code_design
      :link-type: doc

      Want to implement your own methods? Use our template base classes for an easy setup.

   .. grid-item-card:: API Reference
      :link: api
      :link-type: doc

      Full documentation of every module, class, and function.


.. toctree::
   :maxdepth: 1
   :caption: Project
   :hidden:

   installation
   quickstart
   Concepts <concepts>
   Code design <code_design>
   notebooks/index
   API <api>
   contributing
   changelog
   references