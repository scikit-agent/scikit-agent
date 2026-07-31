# Model Analysis and Visualization

These modules extract structured metadata from block models, analyze their
decision structure, and render them as plate-notation diagrams.

## Model Analyzer

```{eval-rst}
.. automodule:: skagent.model_analyzer
   :members:
```

## Model Visualizer

```{eval-rst}
.. automodule:: skagent.model_visualizer
   :members:
```

## Strategic Relevance

Analysis of strategic reliance between decisions, using the Koller & Milch
s-reachability criterion, and the resulting relevance graph. The order a
relevance graph implies, {meth}`~skagent.relevance.RelevanceGraph.condensation`,
is what {class}`skagent.algos.best_response.TabularBestResponseSolver` solves a
block in; see {doc}`algorithms`.

```{eval-rst}
.. automodule:: skagent.relevance
   :members:
```
