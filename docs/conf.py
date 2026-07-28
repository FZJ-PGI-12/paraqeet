import os
from importlib.metadata import version

from sphinx.domains.python import PyObject
from sphinx_pyproject import SphinxConfig

project = "ParaQeet"
__version__ = version("paraqeet")
release = __version__
config = SphinxConfig("../pyproject.toml", globalns=globals(), config_overrides={"version": __version__})

# Install package in editable mode so autodoc can import modules

extensions = [
    "sphinx.ext.napoleon",  # to parse numpy style python docstrings
    "sphinx.ext.mathjax",  # to include math expressions in the .rst files
    "nbsphinx",  # to include jupyter notebooks,
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.autodoc",  # Core library for html generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables
    "sphinx.ext.intersphinx",  # Cross-link to NumPy/SciPy/JAX/Python docs
    "myst_parser",  #  Include md in html
    "sphinx.ext.linkcode",  # To add a source button to each class
    "sphinx_design",  # Grid cards on the landing page
    "sphinx_copybutton",  # Copy-to-clipboard buttons on code blocks
    "sphinxext.opengraph",  # Social media / description metadata
    "sphinxcontrib.bibtex",  # bibliography
]

# Strip console and doctest prompts when copying code blocks
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# OpenGraph metadata for link previews
ogp_site_url = "https://paraqeet.readthedocs.io/en/latest/"
ogp_image = "https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/raw/main/big_logo.png"
ogp_description_length = 200
ogp_site_name = "ParaQeet - A quantum optimal control toolkit with simple parameter management"


# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"  # Removes repeated class definition in __init__

add_module_names = False  # For cleaner documentation with shorter headings

# For documentation of class attributes (similar to arguments)
napoleon_use_ivar = True
# The :ivar: fields render under a hardcoded "Variables" heading — relabel it to "Attributes".
for field in PyObject.doc_field_types:
    if "ivar" in field.names:
        field.label = "Attributes"

autoclass_content = "class"  # to not have repeated documentation for a class and its init

# Make external types (numpy, scipy, jax, python builtins) clickable in the
# rendered API documentation.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}

# Bibliography configurations
bibtex_bibfiles = ["./refs.bib"]  # required; path relative to the source dir
bibtex_default_style = "unsrt"  # or "plain", "alpha", "unsrtalpha"
bibtex_reference_style = "label"  # or "label" (default), "super", "foot"

exclude_patterns = ["**.ipynb_checkpoints"]

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {
        "alt_text": "ParaQeet",
        "text": "ParaQeet",
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo_dark.png",
    },
    "icon_links": [
        {
            "name": "GitLab",
            "url": "https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet",
            "icon": "fa-brands fa-gitlab",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/paraqeet/",
            "icon": "fa-brands fa-python",
        },
    ],
    "switcher": {
        "json_url": "https://paraqeet.readthedocs.io/en/latest/_static/versions.json",
        "version_match": os.environ.get("READTHEDOCS_VERSION", "dev"),
    },
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_nav_level": 1,
    "show_toc_level": 1,
    "show_prev_next": True,  # Enable prev/next buttons
    "collapse_navigation": True,
    "header_links_before_dropdown": 5,
    "navbar_persistent": ["search-button"],
    "show_version_warning_banner": True,
    "secondary_sidebar_items": ["page-toc"],  # show subheadings in sidebar
}


html_title = f"{project} v{release}"
htmlhelp_basename = "paraqeet"
html_favicon = "_static/logo.png"

# Disable “View page source” link for index page
html_show_sourcelink = False

# XeLaTeX for better support of unicode characters
latex_engine = "xelatex"

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]


def linkcode_resolve(domain, info):
    """Resolve link for source code"""
    if domain != "py" or not info["module"]:
        return None

    import importlib
    import inspect

    mod = importlib.import_module(info["module"])
    obj = mod
    for part in info["fullname"].split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        filepath = inspect.getfile(obj)
        source, start_line = inspect.getsourcelines(obj)
    except (TypeError, OSError):
        return None

    filepath = filepath.split("paraqeet/")[-1]
    end_line = start_line + len(source) - 1

    return (
        f"https://jugit.fz-juelich.de/pgi-12-external/qfc/paraqeet/-/blob/main/src/"
        f"paraqeet/{filepath}#L{start_line}-L{end_line}"
    )
