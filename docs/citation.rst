Citation
========

If you use this software, please cite it.

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.23208578.svg
   :target: https://doi.org/10.5281/zenodo.23208578
   :alt: Zenodo DOI


Author list, ORCIDs, and the associated references are maintained in
:download:`CITATION.cff <../CITATION.cff>` at the repository root.
The badge above always resolves to the latest archived release. 
To cite a specific version instead, for example for reproducibility, use that version's own DOI.

Reference manager export
------------------------

Copy the entry that matches your workflow into your ``.bib`` file or reference
manager.

.. tab-set::

   .. tab-item:: BibTeX

      .. code-block:: bibtex

         @software{mishra_2026_23208578,
           author       = {Mishra, Ashutosh and
                           Simm, Alexander and
                           Wald, Moritz and
                           Westphal, Lidia and
                           Ciani, Alessandro and
                           Wittler, Nicolas},
           title        = {ParaQeet - A quantum optimal control toolkit
                           with simple parameter management
                          },
           month        = oct,
           year         = 2026,
           publisher    = {Zenodo},
           version      = {v0.11.1},
           doi          = {10.5281/zenodo.23208578},
           url          = {https://doi.org/10.5281/zenodo.23208578},
         }

   .. tab-item:: RIS

      .. code-block:: text

         TY  - GEN
         AU  - Mishra, Ashutosh
         AU  - Simm, Alexander
         AU  - Wald, Moritz
         AU  - Westphal, Lidia
         AU  - Ciani, Alessandro
         AU  - Wittler, Nicolas
         DA  - 2026-10-07
         DO  - 10.5281/zenodo.23208578
         PY  - 2026
         TI  - ParaQeet - A quantum optimal control toolkit with simple parameter management
         ER  -

   .. tab-item:: EndNote

      .. code-block:: text

         %0 Generic
         %A Mishra, Ashutosh
         %A Simm, Alexander
         %A Wald, Moritz
         %A Westphal, Lidia
         %A Ciani, Alessandro
         %A Wittler, Nicolas
         %D 2026
         %R 10.5281/zenodo.23208578
         %T ParaQeet - A quantum optimal control toolkit with simple parameter management

Zenodo also provides these exports directly on the
`archive record <https://doi.org/10.5281/zenodo.23208578>`_ via *Export ->
Citation / Reference managers*.


Citation File Format
--------------------

The repository ships a `Citation File Format <https://citation-file-format.github.io/>`_
file, :download:`CITATION.cff <../CITATION.cff>`, at its root. GitHub reads it
to show a **"Cite this repository"** button that exports formatted APA and
BibTeX automatically, and reference managers such as Zotero can import it
directly.

To verify the file after editing:

.. code-block:: console

   $ pip install cffconvert
   $ cffconvert --validate -i CITATION.cff

and to regenerate the exports shown above:

.. code-block:: console

   $ cffconvert -f apalike -i CITATION.cff
   $ cffconvert -f bibtex -i CITATION.cff

If a paper describing ParaQeet is published in the future, cite that as the
primary reference and keep the software citation above for the specific release
you used.