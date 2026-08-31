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
over, together with the transforms that pose them:
{meth}`~skagent.influence.SCIM.with_edge`, which adds an information link, and
{meth}`~skagent.influence.SCIM.without_edges`, which drops the links a reduction
finds inert.

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

### Incentive criteria

The same substrate answers a third question, about a node that is neither a
decision nor a shock: what one decision stands to gain from it, or does to it.
These are the four criteria of Everitt, Carey, Langlois, Ortega & Legg, "Agent
Incentives: A Causal Perspective" (AAAI-21; arXiv:2102.01685) --

- {func}`~skagent.relevance.admits_voi`: would observing the node raise the
  achievable payoff?
- {func}`~skagent.relevance.admits_ri`: does every optimal policy respond to a
  change in it?
- {func}`~skagent.relevance.admits_voc`: would setting it raise the achievable
  payoff?
- {func}`~skagent.relevance.admits_ici`: does the decision reach its payoff
  _through_ it?

Each is sound and complete for a diagram holding exactly one decision, and a
diagram with more is refused rather than answered. Three of the four run over
the {func}`~skagent.relevance.minimal_reduction`, the diagram with every
observation {func}`~skagent.relevance.is_requisite` rejects unwired.

The answers are properties of the graph, not of a calibration: `False` means the
incentive is absent under every parameterization the diagram admits, and `True`
means some parameterization has it, not that yours does.

```{eval-rst}
.. automodule:: skagent.relevance
   :members:
```
