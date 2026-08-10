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

## Influence Diagrams

The structural view of a block -- chance, decision and utility nodes with causal
edges -- and the d-separation engine the graphical criteria below are posed
over.

```{eval-rst}
.. automodule:: skagent.influence
   :members:
```

## Relevance

What a decision must still account for, given what it already knows. One
d-separation test, read two ways: across decisions it is the Koller & Milch
s-reachability criterion, and the order the resulting relevance graph implies,
{meth}`~skagent.relevance.RelevanceGraph.condensation`, is what
{class}`skagent.algos.best_response.TabularBestResponseSolver` solves a block in
(see {doc}`algorithms`). Run from a shock instead, it tells a solver whether to
grid that shock or integrate it inside the maximization.

```{eval-rst}
.. automodule:: skagent.relevance
   :members:
```
