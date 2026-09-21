# Loss Functions

The loss module provides the objective functions minimized during neural network
training. Each loss wraps a `BellmanPeriod` and encodes a different optimality
criterion: static reward maximization, estimated discounted lifetime reward,
Bellman equation residuals, or Euler equation residuals.

```{eval-rst}
.. automodule:: skagent.loss
   :members:
```
