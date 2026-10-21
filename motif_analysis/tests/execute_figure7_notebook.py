"""Execute Notebook 01 through IPython when a Jupyter kernel cannot launch.

Run from the project root. Saves actual rich outputs and a standalone HTML
preview; exceptions in any cell stop execution. This uses the same IPython
execution engine as notebook code cells, without a kernel socket connection.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from nbconvert import HTMLExporter
from traitlets.config import Config


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    path = root / "01_connectome_qc_and_motif_census.ipynb"
    notebook = nbformat.read(path, as_version=4)
    shell = InteractiveShell.instance(config=Config({"HistoryManager": {"hist_file": ":memory:"}}))
    shell.user_ns["__name__"] = "__main__"
    number = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        number += 1
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        error = result.error_before_exec or result.error_in_exec
        if error:
            raise RuntimeError(f"Notebook cell {number} failed: {error}") from error
        cell.execution_count = number
        cell.outputs = []
        for name, content in (("stdout", captured.stdout), ("stderr", captured.stderr)):
            if content:
                cell.outputs.append(nbformat.v4.new_output("stream", name=name, text=content))
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata))
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    html, _ = HTMLExporter().from_notebook_node(notebook)
    preview = root / "outputs/consolidated_01_qc_census/rees_figure7/notebook_preview.html"
    preview.write_text(html, encoding="utf-8")
    print(f"Executed {number} code cells; saved notebook and {preview.name}.")


if __name__ == "__main__":
    main()
