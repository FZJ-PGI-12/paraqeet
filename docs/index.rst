.. image:: ../big_logo.png
   :align: center
   :width: 90%
   :alt: logo

===================================================================================
ParaQeet -  A quantum optimal control toolkit with simple parameter management
===================================================================================

Choose a pulse parametrization, simulate a quantum system, and optimize.

ParaQeet combines quantum optimal control methods with automatic differentiation
via JAX, aimed at resource efficient computation. Currently implemented
optimization methods:

- GRAPE: Gradient Ascent Pulse Engineering
- GOAT: Gradient Optimization of Analytic conTrols
- dCRAB: (Gradient based) dressed Chopped RAndom Basis
- GOAToverGRAPE: A variant of GROUP that optimizes continuous pulse parameters with GRAPE inside.

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      Install ParaQeet from PyPI or set up a development environment.

   .. grid-item-card:: Quickstart
      :link: quickstart
      :link-type: doc

      Your first optimization in five minutes: a qubit flip with GOAT.

   .. grid-item-card:: Examples
      :link: notebooks/index
      :link-type: doc

      Notebooks from single-qubit gates to bosonic state preparation.

   .. grid-item-card:: API Reference
      :link: api
      :link-type: doc

      Full documentation of every module, class, and function.

Getting Started
===============

.. toctree::
   :maxdepth: 1

   installation
   quickstart
   Concepts <concepts>

Examples
========

.. toctree::
   :maxdepth: 1

   notebooks/index


API Reference
=============

.. toctree::
   :maxdepth: 1

   API <api>


.. toctree::
   :maxdepth: 1
   :caption: Project
   :hidden:

   contributing
   changelog

More
====

- :doc:`contributing`
- :doc:`changelog`
