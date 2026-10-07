# Contributing to ParaQeet

We warmly welcome all contributions. Please read through this guide before opening a merge request.

## Development setup

Clone the repository and `cd` into the folder, e.g. `cd paraqeet`:

```bash
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -e ."[dev]"
```

## Committing changes

Set up the pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

The repository ships a [pre-commit configuration](.pre-commit-config.yaml) that runs the following checks on every commit - 

- **ruff** (linter) — checks and auto-fixes style issues, using the settings from `pyproject.toml` and including import sorting.
- **ruff-format** (formatter) — formats the code so the whole codebase stays consistent.
- **mypy** — runs static type checking to catch type errors early.
- **nbstripout** — removes output and metadata from Jupyter notebooks, keeping them clean in git history.
- **codespell** — catches common spelling mistakes in code, comments, and docs.

We recommend the contributors to use the pre-commit to follow a consistent code style.

## AI usage policy

We're happy for you to use AI tools while working on ParaQeet — we do too! To keep things transparent and high quality, we ask a couple of small things:

- Please verify and test the code yourself before submitting, even if an AI helped write it.
- Kindly note in your merge request where AI was used.

We'd prefer that commits be created by you, not by an AI, so that the history reflects the human decisions behind them.

## Best practices

### General

- The target user is a scientific person with programming skills.
- Please avoid "let's fix that later" — small cleanups are easier now than later.
- Prefer `jax.numpy` (`import jax.numpy as jnp`) over `numpy` where applicable.
- Write unit tests for (almost) everything. Test for properties instead of specific values. Use dummy classes instead of fixtures whenever possible.
- Explicitly refer to literature in comments when possible.

### Code-specific

- Follow the [PEP 8 naming conventions](https://peps.python.org/pep-0008/#naming-conventions).
- Add specific types to function parameters and return values. Every function must have a return value, even if it is `None`.
- When storing `self.something` as a class variable, declare `something` at the top level of the class.
- Make functions and variables private unless they need to be public.
- Use as few global functions as possible. Instead, put the function into a class and use inheritance.
- For optional parameters, use the type `X | None` instead of `Optional[X]`.

### Documentation

- Add a docstring to every class and function, unless the function inherits the documentation from another class or interface.
- Use the google format for arguments and return values (see [google docstring format](https://google.github.io/styleguide/pyguide.html)).
- For all merge requests involving changes to the example notebooks, compile the examples for documentation:
  ```bash
  pip install matplotlib nbconvert ipykernel pandoc
  ```
  ```bash
  for notebook in examples/*.ipynb; do jupyter nbconvert --config docs/nbconvert_config.py --execute --to rst --output-dir docs/notebooks $notebook; done
  ```
- To test the build locally:
  ```bash
  pip install -r docs/requirements.txt
  ```
  ```bash
  sphinx-build docs/ docs/_build -W
  ```
- To update the requirements in `docs/`:
  ```bash
  pip install pip-tools
  ```
  Update `requirements.in` and then:
  ```bash
  pip-compile --no-strip-extras --output-file=requirements.txt requirements.in
  ```

### Changelog

- Keep an updated `Unreleased` section during the merge request.
- At the time of release, change `Unreleased` to the version number.
- Add the version number with a `v`. E.g., `v0.1.0`.
- Avoid dumping the git log into the changelog.
- Add a comparison link at the end of the document.
- Add the date in year-month-day format.
- Follow the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) guide for details about formatting.
