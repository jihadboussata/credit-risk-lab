"""A failed notebook must not be published with invented or partial outputs."""
import importlib.util
from pathlib import Path
import nbformat
import pytest


def test_failed_execution_preserves_original_file(tmp_path):
    path = tmp_path / "fails.ipynb"
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("raise ValueError('sentinel')")])
    nbformat.write(notebook, path)
    original = path.read_bytes()
    runner = Path(__file__).resolve().parents[1] / "scripts/execute_notebooks.py"
    spec = importlib.util.spec_from_file_location("notebook_executor", runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(RuntimeError, match="sentinel"):
        module.execute_notebook(path, tmp_path)
    assert path.read_bytes() == original
