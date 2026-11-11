# C kernels

The optional compiled kernels (`native/`), documented from the comment
before each declaration. `wm_kernels.c` is compiled once per instruction-set
tier with `WM_TIER` set, and every `WM_FN(name)` becomes `name_<tier>`; the
reference shows each under its portable spelling. `wm_dispatch.c` picks a
tier when the library loads and exports the `wm_*` entry points that
`weathermentations._native` calls through `ctypes`. Each kernel's Python
caller, and the numpy code it stands in for, is named in its comment.

## The kernel table

```{eval-rst}
.. c:autodoc:: wm_kernels.h
```

## The kernels

```{eval-rst}
.. c:autodoc:: wm_kernels.c
```

## Dispatch and the exported entry points

```{eval-rst}
.. c:autodoc:: wm_dispatch.c
```

## The cross-machine hash check

A program of its own, so its names are in a namespace of their own.

```{eval-rst}
.. c:namespace-push:: hash_check

.. c:autodoc:: check/wm_hashes.c

.. c:namespace-pop::
```

## Plain C against intrinsics

A program of its own: the library's three loops written twice, timed and
compared. It is compiled for AVX2 on x86-64 and for NEON on arm64, so the
reference shows whichever the documentation was built on.

```{eval-rst}
.. c:namespace-push:: simd_experiment

.. c:autodoc:: experiments/simd_explicit.c

.. c:namespace-pop::
```
