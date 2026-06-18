# Configuration file for the Sphinx documentation builder.

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("../.."))

project = "analyzerl_parser"
copyright = "2026, Owen Singh"
author = "Owen Singh"

pyproject_text = Path("../../pyproject.toml").read_text(encoding="utf-8")
release = re.search(r'^version = "([^"]+)"', pyproject_text, re.MULTILINE).group(1)

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

autodoc_mock_imports = [
    "cv2",
    "matplotlib",
    "mpl_toolkits",
    "numpy",
    "pandas",
    "polars",
]

html_theme = "sphinx_rtd_theme"
html_static_path = []
