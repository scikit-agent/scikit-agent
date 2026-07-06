# Models

The `skagent.models` subpackage contains predefined economic models. For a
narrative overview of the benchmark registry and how to call it, see the
{doc}`../user_guide/benchmark_models` guide; for runnable, plotted walkthroughs
see the {doc}`../auto_examples/index`.

## Consumer Models

```{eval-rst}
.. automodule:: skagent.models.consumer
   :members:
```

## Benchmark Registry

The benchmark registry catalogues discrete-time dynamic programming problems
with known closed-form policies (plus a few that are kept for numerical
validation). Its public helpers and analytical-policy functions are listed
below.

```{eval-rst}
.. automodule:: skagent.models.benchmarks
   :no-members:
```

### Registry access

```{eval-rst}
.. autofunction:: skagent.models.benchmarks.list_benchmark_models
.. autofunction:: skagent.models.benchmarks.get_benchmark_model
.. autofunction:: skagent.models.benchmarks.get_benchmark_calibration
.. autofunction:: skagent.models.benchmarks.get_analytical_policy
.. autofunction:: skagent.models.benchmarks.has_analytical_policy
.. autofunction:: skagent.models.benchmarks.get_reference_policy
.. autofunction:: skagent.models.benchmarks.get_test_states
.. autofunction:: skagent.models.benchmarks.get_custom_validation
```

### Validation

```{eval-rst}
.. autofunction:: skagent.models.benchmarks.validate_analytical_solution
.. autofunction:: skagent.models.benchmarks.euler_equation_test
.. autofunction:: skagent.models.benchmarks.get_analytical_lifetime_reward
```

### Analytical policies

```{eval-rst}
.. autofunction:: skagent.models.benchmarks.d1_analytical_policy
.. autofunction:: skagent.models.benchmarks.d2_analytical_policy
.. autofunction:: skagent.models.benchmarks.d3_analytical_policy
.. autofunction:: skagent.models.benchmarks.u1_analytical_policy
.. autofunction:: skagent.models.benchmarks.u2_analytical_policy
.. autofunction:: skagent.models.benchmarks.crra_utility
```

## Fisher Two-Period Model

```{eval-rst}
.. automodule:: skagent.models.fisher
   :members:
```

## Perfect Foresight Models

```{eval-rst}
.. automodule:: skagent.models.perfect_foresight
   :members:
```

## Perfect Foresight Models (Normalized)

```{eval-rst}
.. automodule:: skagent.models.perfect_foresight_normalized
   :members:
```

## Resource Extraction

```{eval-rst}
.. automodule:: skagent.models.resource_extraction
   :members:
```
