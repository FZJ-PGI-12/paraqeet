from importlib.metadata import version

from sphinx_pyproject import SphinxConfig

project = "ParaQeet"
__version__ = version("paraqeet")
release = __version__
config = SphinxConfig("../pyproject.toml", globalns=globals(), config_overrides={"version": __version__})

# Install package in editable mode so autodoc can import modules

extensions = [
    "sphinx.ext.napoleon",  # to parse numpy stye python docstrings
    "sphinx.ext.mathjax",  # to include math expressions in the .rst files
    "recommonmark",  # to include markdown files in sphinx documentation
    "nbsphinx",  # to include jupyter notebooks,
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.autodoc",  # Core library for html generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables
]

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

exclude_patterns = ["**.ipynb_checkpoints"]

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {"image": "../logo.png"},
    "gitlab_url": "https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet",
    "show_nav_level": 1,
    "show_toc_level": 1,
    "show_prev_next": True,  # Enable prev/next buttons
    "collapse_navigation": True,
    "header_links_before_dropdown": 4,
    "navbar_end": ["search-button", "theme-switcher", "version-switcher", "navbar-icon-links"],
    "navbar_persistent": [],
    "show_version_warning_banner": True,
    "secondary_sidebar_items": ["page-toc"],  # show subheadings in sidebar
}


html_title = f"{project} v{release}"
htmlhelp_basename = "paraqeet"

# Disable “View page source” link for index page
html_show_sourcelink = False

# XeLaTeX for better support of unicode characters
latex_engine = "xelatex"

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
