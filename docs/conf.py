"""Sphinx configuration for the BCRFoundation tutorials."""

from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]

project = metadata["name"]
author = "Haotian Zhuang"
copyright = "2026, Haotian Zhuang"
release = metadata["version"]
version = release

extensions = ["myst_parser", "sphinx_copybutton"]
source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "README.md"]
myst_heading_anchors = 3
nitpicky = True

html_theme = "furo"
html_title = "BCRFoundation Documentation"
html_theme_options = {
    "source_repository": "https://github.com/haotian-zhuang/BCRFoundation/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#157d85",
        "color-brand-content": "#116b73",
    },
    "dark_css_variables": {
        "color-brand-primary": "#6cc9ca",
        "color-brand-content": "#8bd9d8",
    },
}
html_show_sourcelink = True
