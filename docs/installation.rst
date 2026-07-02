.. _install:

Installation
============

ParaQeet requires **Python 3.12 or newer**.

From PyPI
---------

To use ParaQeet, install it from PyPI with pip:

.. code-block:: console

   $ pip install paraqeet

We recommend installing into an isolated environment, e.g. with ``virtualenv``:

.. code-block:: console

   $ pip install virtualenv
   $ virtualenv venv
   $ source venv/bin/activate
   $ pip install paraqeet

From source (development install)
---------------------------------

To work on ParaQeet itself, clone the repository and install it in editable mode together with the
development dependencies (``pytest``, ``ruff`` and ``pre-commit``):

.. code-block:: console

   $ git clone https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet.git
   $ cd paraqeet
   $ pip install -e .[dev]

If you contribute changes, we recommend to install the pre-commit hooks (ruff lint, ruff-format, mypy, codespell and
nbstripout run on every commit):

.. code-block:: console

   $ pip install pre-commit
   $ pre-commit install