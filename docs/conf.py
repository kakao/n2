# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if not on_rtd:
    sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'N2'
copyright = '2017, Kakao Corp.'
author = 'Kakao Recommendation Team'

master_doc = 'index'

# The full version, including alpha/beta/rc tags
release = '0.1.7'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',  # include documentation from docstrings
    'sphinx.ext.napoleon',  # support NumPy and Google style docstrings
    'breathe',  # support cpp documentation
    'exhale',  # support cpp documentation
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax'
]

# Napoleon settings
napoleon_include_init_with_doc = True

# Setup the breathe extension
breathe_projects = {
    "N2": "./doxyoutput/xml"  # tell breathe that doxygen xml output file can be found here.
}
breathe_default_project = "N2"

# Setup the exhale extension
exhale_args = {
    # These arguments are required
    "containmentFolder": "./api",  # exhale will produce docs/api folder
    "rootFileName": "cpp_reference_root.rst",  # exhale will produce docs/api/library_root.rst
    "rootFileTitle": "CPP Reference",
    "doxygenStripFromPath": "..",
    # Suggested optional arguments
    "createTreeView": True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin": "INPUT = ../include"
    # it would use doxygen to parse ../include and save the output to docs/doxyoutput
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'README.rst']

smartquotes = False  # deactivate smartquotes

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'custom.css'
]
