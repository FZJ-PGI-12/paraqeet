===================================================================================
paraQeet -  A quantum optimal control toolkit with parameter handling
===================================================================================

You can find quick information on :ref:`installation <install>` and contributing in the `README`_ and `CONTRIBUTING`_ documents.

Choose a pulse parametrisation, simulate a quantum system, and optimise.

Combining Quantum Optimal Control methods with automatic differentiation with JAX.
Aimed at resource efficient computation.

We use a top-down approach to make the codebase modular.
Each module interacts only with the module above it in hierarchy.

.. _README: https://jugit.fz-juelich.de/pgi-12-external/yaq/yaq/-/blob/main/README.md
.. _CONTRIBUTING: https://jugit.fz-juelich.de/pgi-12-external/yaq/yaq/-/blob/main/CONTRIBUTING.md

.. image:: layers.png
   :width: 400
   :alt: layers

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Examples:
   :glob:
   
   notebooks/*

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API:
   :glob:
   
   source/*


Contents
========
.. toctree::
   
   usage



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
