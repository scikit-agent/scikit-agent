# Roadmap

The current version of `scikit-agent`, v0.1, is a proof of concept for a much
more ambitious raodmap.

Here, we will briefly summarize the envisioned scope of this library. You can
see our [issue tracker](https://github.com/scikit-agent/scikit-agent/issues) for
a complete list of open issues.

## Broader model support

The current algorithm implementations have very limited assumptions. We support
only a single agent type, do not support agent interaction in simulation, and
support only one control variable.

We believe our framework is extensible to many more causal, multi-agent
environments and plan to do so in future releases.

- **Solution methods for multiple controls per agent.** We currently allow
  models to have multiple controls per agent, but do not yet fully support
  solution algorithms that solve for multiple controls simultaneously. We will.
- **Agent interaction**. We do not yet support agent interaction, though this is
  essential for e.g. macroeconomic modeling. In future implementations, we will
  enable agents to interact both interpersonally and through aggregate
  structures, such as markets.
- **Multiple agent roles**. We will support models with agents varying widely in
  roles, including different reward and action spaces.
- **Strategic equilibrium solvers**. Efficiently solving for strategic
  equilibrium between multiple agents.

As we expand the classes of models that `scikit-agent` can handle, we will also
provide a larger library of example models that showcase these features.

## More algorithms

We have launched `scikit-agent` with the Maliar algorithm as its flagship
solution algorithm.

We aim to include a wider range of solution algorithms in future releases,
including model-free reinforcement learninig algorithms like Deep Deterministic
Policy Gradient (DDPG) as well as classic dynamic programming methods and
evolutionary methods

Our goal is for `scikit-agent` to ultimately be a tool for testing the
performance of many different algorithms on causal agent models.

## Inference functions

We aim to provide better support for using `scikit-agent` models to make
inferences from data. Currently, we support Monte Carlo simulation from
empirically calibrated models, facilitating forecasting. In the future, we will
support:

- **Structural estimation**. Given an empirical dataset that includes emergent
  properties of the model, we will provide ways of estimating the parameters of
  the model which make the empirical targets likely. We have in mind Approximate
  Bayesian Computation as well as deep-learning based techniques.
- **Model selection** Given an empirical dataset, we will include functions that
  support _model selection_, the search for the model structure that makes the
  target data most likely.

Ultimately, we would like `scikit-agent` to be a powerful tool for inferring and
representing causal multi-agent models from empirical data, and reasoning about
the implications of these models.
