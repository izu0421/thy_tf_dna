#!/usr/bin/env python3
"""Build the project site from scripts/site/page.html.

page.html is the page body as published to the claude.ai artifact (title, style, markup, script; no
<html>/<head> wrapper). This writes it as a standalone document to docs/index.html and, because GitHub
Pages serves this repository from its root, to /index.html with paths pointing into docs/.
The structure files in docs/s/ come from scripts/site/export_structures.py.

Usage (from the project root): python scripts/site/build_site.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
src = (ROOT / "scripts" / "site" / "page.html").read_text()
cut = src.index("</style>") + len("</style>")
head, body = src[:cut], src[cut:]
head = head.replace('<meta name="claude:viewport-fit-sides" content="cover">\n', '')
old = '· both views move together</span>'
assert body.count(old) == 1
body = body.replace(old, '· both views move together · <a href="overview.html">full write-up</a></span>')
doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
       '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
       '<meta name="description" content="AlphaFold 3 models of RUNX1 with LEF1 and ETS1 on high- and '
       'low-syntax DNA, side by side in 3D.">\n' + head + "\n</head>\n<body>\n" + body + "\n</body>\n</html>\n")
(ROOT / "docs" / "index.html").write_text(doc)
assert doc.count('"file":"s/') == 4 and doc.count('href="overview.html"') == 1
root = doc.replace('"file":"s/', '"file":"docs/s/').replace('href="overview.html"', 'href="docs/overview.html"')
(ROOT / "index.html").write_text(root)
(ROOT / ".nojekyll").touch()
print("built docs/index.html and index.html")
