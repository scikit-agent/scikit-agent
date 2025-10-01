from __future__ import annotations

import importlib.metadata
from typing import Any

project = "scikit-agent"
copyright = "2025, scikit-agent Team"
author = "scikit-agent Team"
version = release = importlib.metadata.version("scikit_agent")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
]

# Sphinx-Gallery configuration
sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to example scripts
    "gallery_dirs": "auto_examples",  # path to gallery generated output
    "filename_pattern": "/plot_",  # pattern to match example files
    "ignore_pattern": r"__init__\.py",  # patterns to ignore
    "plot_gallery": "True",  # create a gallery
    "download_all_examples": False,  # don't create zip downloads by default
    "remove_config_comments": True,  # remove config comments from examples
    "expected_failing_examples": [],  # list of examples expected to fail
    "image_scrapers": ("matplotlib",),  # image scrapers
    "first_notebook_cell": "%matplotlib inline",  # first cell for notebooks
    "show_memory": False,  # don't show memory usage
    "show_signature": True,  # show function signatures
    "min_reported_time": 0,  # minimum time to report
    # Note: reference_url was removed to fix "Ran out of input" error during hyperlink embedding
    "backreferences_dir": None,  # disable backreferences to avoid embedding issues
}

source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
    "DOCS_DEPLOYMENT.md",
]

html_theme = "furo"

html_theme_options: dict[str, Any] = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/scikit-agent/scikit-agent",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
    "source_repository": "https://github.com/scikit-agent/scikit-agent",
    "source_branch": "main",
    "source_directory": "docs/",
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

always_document_param_types = True

# HTML settings
html_title = f"{project} {version}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Auto-generated API docs
autosummary_generate = True
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
