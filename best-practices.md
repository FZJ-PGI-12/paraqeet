## General

- The target user is a scientific person with programming skills
- No "let's fix that later"
- Use tensorflow wherever needed. Conversion to numpy is only done in the optimiser.
- Write unit tests for (almost) everything. Test for properties instead of specific values.

## Code-specific

- Add specific types to function parameters and return values. Every function must has a return value, even if it
  is `None`.
- When storing `self.something` as a class variable, specify `something` in the top level of the class.
- Make functions and variables private unless they are needed to be public
- Use as few global functions as possible. Instead, put the function into a class an use inheritance.
- For optional parameters, use the type `X | None` instead of `Optional[X]`

## Documentation

- Add a doc string to every class and function, unless the function inherits the documentation from another class or interface.
- Use the numpy format for arguments and return values (see https://numpydoc.readthedocs.io/en/latest/format.html)
