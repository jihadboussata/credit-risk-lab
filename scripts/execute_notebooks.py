"""Execute the study notebooks and save their genuinely computed rich outputs.

Run from the project root:
    python scripts/execute_notebooks.py
    python scripts/execute_notebooks.py 02_exploration_and_hypotheses

This small IPython runner uses no socket kernel, which also makes it useful in
restricted desktop environments. Every code cell runs in order in a fresh
notebook namespace. A cell failure exits nonzero and leaves that notebook's
previous saved version intact. Source cells and computed values are not edited.
Install the project's [dev] dependencies before running this script.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import sys

import matplotlib
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from matplotlib_inline.backend_inline import configure_inline_support
from IPython.core.pylabtools import select_figure_formats


def execute_notebook(path: Path, root: Path) -> int:
    """Execute a notebook; preserve only real captured outputs and execution metadata."""
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    started = datetime.now(timezone.utc).isoformat()
    original_directory = Path.cwd()
    shell = InteractiveShell.instance()
    matplotlib.use("module://matplotlib_inline.backend_inline")
    configure_inline_support(shell, "inline")
    select_figure_formats(shell, {"png"})
    executed = 0
    try:
        os.chdir(root)
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            with capture_output(stdout=True, stderr=True, display=True) as captured:
                result = shell.run_cell(cell.source, store_history=True)
            error = result.error_before_exec or result.error_in_exec
            if error is not None:
                raise RuntimeError(
                    f"{path.name}, code cell {executed + 1} failed: {type(error).__name__}: {error}"
                ) from error
            outputs = []
            if captured.stdout:
                outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
            if captured.stderr:
                outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
            for output in captured.outputs:
                outputs.append(nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata))
            cell.outputs = outputs
            cell.execution_count = result.execution_count
            executed += 1
            print(f"{path.name}: executed code cell {executed}", flush=True)
        notebook.metadata["execution"] = {
            "method": "IPython in-process execution of real source cells",
            "started_utc": started,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
        }
        notebook.metadata.setdefault("language_info", {})["version"] = platform.python_version()
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        return executed
    finally:
        os.chdir(original_directory)
        import matplotlib.pyplot as plt
        plt.close("all")
        InteractiveShell.clear_instance()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebooks", nargs="*", help="Notebook filenames or stems; default: all study notebooks")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    available = {p.stem: p for p in sorted((root / "notebooks").glob("*.ipynb"))}
    if not available:
        parser.error("No notebooks found under the project root.")
    requested = args.notebooks or list(available)
    paths = []
    for name in requested:
        if Path(name).name != name or Path(name).stem not in available:
            parser.error(f"Unknown notebook {name!r}. Choose from: {', '.join(available)}")
        paths.append(available[Path(name).stem])
    for path in paths:
        count = execute_notebook(path, root)
        print(f"Saved {path.name} ({count} code cells).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
