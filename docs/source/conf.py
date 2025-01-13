# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'BART-Survival'
copyright = '2024, Jacob Tiegs'
author = 'Jacob Tiegs'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 
                'sphinx.ext.coverage', 
                'sphinx.ext.napoleon',
                "nbsphinx",
                "myst_parser"
                ]

templates_path = ['_templates']
exclude_patterns = []

myst_enable_extensions = [
    "dollarmath",
    "amsmath"
]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'classic'
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_theme_options = {
#     # "nosidebar":True
    # "rightsidebar": "true"
}

import os
import sys
sys.path.insert(0, os.path.abspath('../../src/bart_survival'))