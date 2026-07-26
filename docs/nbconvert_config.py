"""nbconvert configuration for building the example documentation.

Pass this file to nbconvert when regenerating the notebook ``.rst`` files::

    jupyter nbconvert --config docs/nbconvert_config.py --execute --to rst \
        --output-dir docs/notebooks examples/<notebook>.ipynb

Its job is to turn the clean citation tags used inside the notebooks

    <cite data-cite="heeres2017implementing">Heeres et al., 2017</cite>

into the visible text followed by a reStructuredText citation role that
sphinxcontrib-bibtex understands

    Heeres et al., 2017 :cite:p:`heeres2017implementing`

The tag renders as plain (italic) text in Jupyter, so the notebooks stay
readable, while the generated ``.rst`` keeps that same text *and* adds a real,
linked citation (e.g. ``(Heeres et al., 2017) [5]``) resolved against
``docs/refs.bib``.
"""

import re

from nbconvert.preprocessors import Preprocessor


class CiteToRstPreprocessor(Preprocessor):
    """Rewrite ``<cite data-cite="key">text</cite>`` tags into the visible text
    plus an rst cite role, so the built docs show both (e.g. ``(text) [5]``).

    The cite role uses pandoc's raw-inline syntax (``...``{=rst}) so it survives
    the markdown-to-rst conversion performed by the rst exporter.
    """

    _CITE = re.compile(r'<cite[^>]*\bdata-cite="([^"]+)"[^>]*>(.*?)</cite>', re.DOTALL)

    def preprocess_cell(self, cell, resources, index):
        if cell.cell_type == "markdown":
            cell.source = self._CITE.sub(
                lambda m: m.group(2).strip() + " ``:cite:p:`" + m.group(1) + "` ``{=rst}",
                cell.source,
            )
        return cell, resources


c.RSTExporter.preprocessors = [CiteToRstPreprocessor]  # noqa: F821
