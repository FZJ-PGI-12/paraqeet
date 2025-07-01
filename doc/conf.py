from sphinx_pyproject import SphinxConfig

config = SphinxConfig("../pyproject.toml", globalns=globals())

copyrights = "PGI-12"
release = "0.9"

extensions = [
    "sphinx.ext.napoleon",  # to parse numpy stye python docstrings
    "sphinx.ext.mathjax",  # to include math expressions in the .rst files
    "recommonmark",  # to include markdown files in sphinx documentation
    "nbsphinx",  # to include jupyter notebooks,
    "sphinx_autodoc_typehints",
]


html_theme = "press"
# html_permalinks_icon = '<span>#</span>'
# html_theme = 'sphinxawesome_theme'

# -- Options for autodoc ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Group modules
autoapi_member_order = "groupwise"

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

