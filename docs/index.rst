.. image:: ../big_logo.png
   :align: center
   :width: 90%
   :alt: logo

===================================================================================
ParaQeet -  A quantum optimal control toolkit with simple parameter management
===================================================================================

You can find quick information on :ref:`installation <install>` and contributing in the `README`_ and `CONTRIBUTING`_ documents.

Choose a pulse parametrisation, simulate a quantum system, and optimise.

Combining Quantum Optimal Control methods with automatic differentiation with JAX.
Aimed at resource efficient computation.

We use a top-down approach to make the codebase modular.
Each module interacts only with the module above it in hierarchy.

.. _README: https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/blob/main/README.md
.. _CONTRIBUTING: https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/blob/main/CONTRIBUTING.md

.. image:: layers.png
   :align: center
   :width: 60%
   :alt: layers


Currently implementated optimization methods
- GRAPE: Gradient Ascent Pulse Enginnering
- GOAT: Gradient Optimization of Analytic conTrols
- dCRAB : (Gradient based) dressed Chopped RAndom Basis

Installation
========
.. toctree::
   
   usage


Examples
========
We introduce the usage of the package with some examples. These are also available as interactive notebooks.

.. toctree::
   :maxdepth: 1
   :glob:

   notebooks/*


API Documentation
=================

.. toctree::
   :maxdepth: 1
   :glob:
   
   source/*


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
