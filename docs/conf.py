import sys
import os 
sys.path.insert(0, os.path.abspath(".."))
from SPACEL import Spoint
from SPACEL import Splane
from SPACEL import Scube
# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'SPACEL'
author = 'Hao Xu'

release = '1.1.3'
version = '1.1.3'

# -- General configuration
exclude_patterns = ['_build', '.DS_Store', '**.ipynb_checkpoints']
extensions = [
    'myst_parser',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'nbsphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'