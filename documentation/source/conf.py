import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'loreleai'
copyright = '2020, Sebastijan Dumancic, Alexander L. Hayes'
author = 'Sebastijan Dumancic, Alexander L. Hayes'

# The full version, including alpha/beta/rc tags
release = '0.0.1'

# -- General configuration ---------------------------------------------------

extensions = [
]

templates_path = ['_templates']

exclude_patterns = ["build"]


# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'

html_static_path = ['_static']
