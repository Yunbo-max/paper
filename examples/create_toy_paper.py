"""A tiny, explicitly synthetic paper PDF for testing the toolchain; not a real research claim."""
from __future__ import annotations
import sys

try:
    import pymupdf
except ImportError as exc:
    raise SystemExit("Install PyMuPDF: pip install 'paperrev-lab[pdf]'") from exc

if len(sys.argv) != 2:
    raise SystemExit("Usage: python examples/create_toy_paper.py /tmp/toy.pdf")
doc = pymupdf.open()
page = doc.new_page()
page.insert_text((60, 50), "Toy Paper: A deterministic transformation")
page.insert_text((60, 90), "Every element is multiplied by two.")
doc.save(sys.argv[1])
doc.close()
print(sys.argv[1])
