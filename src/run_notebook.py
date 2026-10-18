"""Execute the research notebook with this interpreter and retain its outputs."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
for key, value in {
    'MPLCONFIGDIR': '/tmp/kinyanews-matplotlib',
    'IPYTHONDIR': '/tmp/kinyanews-ipython',
    'JUPYTER_CONFIG_DIR': '/tmp/kinyanews-jupyter',
    'JUPYTER_DATA_DIR': '/tmp/kinyanews-jupyter-data',
    'JUPYTER_RUNTIME_DIR': '/tmp/kinyanews-jupyter-runtime',
    'OPENBLAS_NUM_THREADS': '2',
    'OMP_NUM_THREADS': '2',
}.items():
    os.environ.setdefault(key, value)

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    path = ROOT / 'notebooks/research_walkthrough.ipynb'
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(notebook, km=manager, timeout=3600,
                            resources={'metadata': {'path': str(ROOT)}})
    client.on_cell_start = lambda cell, cell_index: print(f'Executing cell {cell_index}', flush=True)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
        manager.shutdown_kernel(now=True)
    print(f'Notebook executed and saved: {path}', flush=True)


if __name__ == '__main__':
    main()
