# Notebook-Centered Repo

This repo has been reduced to the notebooks that are still in use plus the local Python modules and datasets they depend on.

## Kept Structure

- `notebooks/`: active analysis notebooks
- `src/finance_data/`: data loaders and Sharpe/DSR helpers used by the notebooks
- `src/sharpe_mc.py`: Monte Carlo helpers used by `experiment_bsc.ipynb`
- `data/`: tracked datasets used directly by the notebooks
- `notebooks/outputs/experiment_bsc/`: retained notebook output cache for the BSC experiment notebooks

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## UCloud Setup

`pip install -e .` only works when your current directory is the project root (the directory that contains `pyproject.toml`).
If you run it from `/work` and the repo is in a subdirectory, installation fails.

Use an absolute repo path on UCloud:

```bash
find /work -maxdepth 4 -name pyproject.toml
python3 -m pip install -e /absolute/path/to/repo
python3 -m ipykernel install --user --name rafsbsc --display-name "Python (rafsbsc)"
```

In the notebook, select kernel `Python (rafsbsc)` and verify:

```python
import sys
print(sys.executable)
```

The notebooks still rely on live Ken French downloads for any dataset that is not already present under `data/`.
