"""
cd vapc/docs
make.bat html
start build/html/index.html
"""


import os
import vapc
import subprocess
import sys



sys.path.insert(0, os.path.abspath("../../src"))
os.environ["XDG_DATA_DIRS"] = os.path.abspath("../tests/data")

# Configuration file for the Sphinx documentation builder.

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "vapc"
copyright = "2024, 3DGeo Research Group, Heidelberg University"
author = "Ronald Tabernig"
release = vapc.get_version()

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # For NumPy and Google style docstrings
    'sphinx_autodoc_typehints',  # For type hints
    'sphinx.ext.viewcode',  # To include source code
    "nbsphinx",
    "nbsphinx_link",
    "sphinx_mdinclude",
    "sphinx_rtd_theme"
]

templates_path = []
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []

napoleon_google_docstring = False
napoleon_numpy_docstring = True
