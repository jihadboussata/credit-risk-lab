"""Validate the committed notebooks without network access or training."""
from pathlib import Path
import nbformat

root = Path(__file__).resolve().parents[1]
paths = sorted((root / "notebooks").glob("*.ipynb"))
if not paths:
    raise SystemExit("No notebooks found")
for path in paths:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    errors = [output for cell in notebook.cells if cell.cell_type == "code"
              for output in cell.get("outputs", []) if output.output_type == "error"]
    if errors:
        raise SystemExit(f"Stored execution error in {path.name}")
    print(f"Validated {path.name}")
