# Scripts, build and CI

What runs around the library: the build hook that compiles the kernels, the
benchmarks and diagnostics in `scripts/`, the generators of the example
images in `docs/`, the shell scripts that check and time the kernels, and
the continuous-integration workflows.

## The build hook

```{eval-rst}
.. automodule:: hatch_build
```

## Benchmarks and diagnostics

```{eval-rst}
.. automodule:: benchmark
```

```{eval-rst}
.. automodule:: benchmark_single_core
```

```{eval-rst}
.. automodule:: benchmark_pipelines
```

```{eval-rst}
.. automodule:: diagnose
```

## The example images

```{eval-rst}
.. automodule:: generate_examples
```

```{eval-rst}
.. automodule:: generate_scene_examples
```

```{eval-rst}
.. automodule:: gallery
```

## Shell scripts

```{script} native/check/run.sh
```

```{script} native/experiments/run.sh
```

## Continuous integration

```{script} .github/workflows/ci.yml
```

```{script} .github/workflows/gallery.yml
```

```{script} .github/workflows/publish.yml
```

## This site

```{eval-rst}
.. automodule:: scriptdoc
```
