from importlib.metadata import version

from sphinx_pyproject import SphinxConfig

__version__ = version("paraqeet")
release = __version__
config = SphinxConfig("../pyproject.toml", globalns=globals(), config_overrides = {"version": __version__})



extensions = [
    "sphinx.ext.napoleon",  # to parse numpy stye python docstrings
    "sphinx.ext.mathjax",  # to include math expressions in the .rst files
    "recommonmark",  # to include markdown files in sphinx documentation
    "nbsphinx",  # to include jupyter notebooks,
    "sphinx.ext.apidoc",
    "IPython.sphinxext.ipython_console_highlighting",
]

exclude_patterns = ["**.ipynb_checkpoints"]


html_theme = "press"

# html_permalinks_icon = '<span>#</span>'
# html_theme = 'sphinxawesome_theme'

# -- Options for autodoc ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Group modules
apidoc_module_first = True
autoapi_member_order = "groupwise"

apidoc_modules = [{"path": "../paraqeet", "destination": "source/"}]

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

html_static_path = ["_static"]
html_css_files = ["custom.css"]
