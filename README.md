# hiaac-librep


## Installation

To install `librep`, you may use:

```
pip install git+https://github.com/otavioon/hiaac-librep.git
```

### Pip optional dependencies

- `deep-learning`: install packages to run deep learning models
- `dev`: install development packages
- `docs`: install packages to build documentation

## Contributing

In order to contribute with `librep` you may want to:

1. Clone librep's repository:

```
git clone https://github.com/otavioon/hiaac-librep.git
```

2. Create a python virtual environment and activate it (requires Python >= 3.8):

```
cd hiaac-librep
python -m venv .librep-venv
source .librep-venv/bin/activate
```

3. Install librep development packages, in editable mode

```
pip install -e .[dev]
```

4. Run tests

```
pytest
```
