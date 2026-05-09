# Setup

## One-time environment setup

```bash
# From the project root
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name hazelnut --display-name "Python (hazelnut)"
```

## Running the notebook

```bash
.venv/bin/jupyter notebook notebooks/hazelnut_basket_eda.ipynb
```

Then select kernel **Python (hazelnut)** in the top-right if it isn't already set.

## Notes

- The `.venv` is already created and the kernel is already registered — you only need to run the one-time setup again if you recreate the environment from scratch.
- All data files are in `data/raw/`. The notebook reads them with relative paths (`../data/raw/`), so always launch Jupyter from the project root.
- ERA5 downloads require a CDS API key in `~/.cdsapirc`. The notebook doesn't re-download ERA5 data — it reads the pre-built `data/raw/hazelnut_master_with_spot.csv`.
