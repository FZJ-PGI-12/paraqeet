# Install the package in developer mode

Clone the repository and cd into the folder, e.g. `cd paraqeet`

```
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -e .[dev]
```

# Commiting changes

Setup pre-commit hooks

```
pip install pre-commit
pre-commit install
```

# Best practices

## General

- The target user is a scientific person with programming skills
- No "let's fix that later"
- Use the numpy package of jax (`import jax.numpy as jnp`) instead of numpy, when applicable.
- Write unit tests for (almost) everything. Test for properties instead of specific values. Use dummy classes instead of fixtures whenever possible.
- Explicitly refer to literature in comments when possible

## Code-specific

- Follow the [PEP 8 naming conventions](https://peps.python.org/pep-0008/#naming-conventions)
- Add specific types to function parameters and return values. Every function must has a return value, even if it
  is `None`.
- When storing `self.something` as a class variable, specify `something` in the top level of the class.
- Make functions and variables private unless they are needed to be public
- Use as few global functions as possible. Instead, put the function into a class an use inheritance.
- For optional parameters, use the type `X | None` instead of `Optional[X]`

## Documentation

- Add a doc string to every class and function, unless the function inherits the documentation from another class or interface.
- Use the numpy format for arguments and return values (see https://numpydoc.readthedocs.io/en/latest/format.html)
- For all merge requests involving changes to the example notebooks, run the following commands to compile the examples for documentation.
  ```
    pip install matplotlib nbconvert ipykernel pandoc
  ```
  ```bash
    for notebook in examples/*.ipynb; do jupyter nbconvert --execute --to rst --output-dir docs/notebooks $notebook; done
  ```