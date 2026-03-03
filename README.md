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

The notebooks still rely on live Ken French downloads for any dataset that is not already present under `data/`.
