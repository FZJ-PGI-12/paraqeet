# Install the package in developer mode

Clone the repository and cd into the folder, e.g. `cd c3-update`

```
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install pytest
pip install -r requirements.txt
pip install -e .
```

# Commiting changes

Setup pre-commit hooks

```
pip install pre-commit
pre-commit install
```