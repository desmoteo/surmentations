# The test suite

What each test module holds to, and every test in it. A test's name is the
claim it checks, written as a sentence, so tests carry a docstring only
where the name leaves something out. The helpers and fixtures each module
builds its cases from are documented with it.

Run it with `pytest tests -q`; `SURMENTATIONS_FAST=0` runs it in exact
mode, `SURMENTATIONS_NATIVE=0` without the compiled kernels.

```{toctree}
:maxdepth: 1
:glob:

tests/*
```
