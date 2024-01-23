"""Sphinx configuration."""
project = "Graph-COMPASS"
author = "Mayar Ali and Merel Kuijs"
copyright = "2024, Mayar Ali and Merel Kuijs"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
