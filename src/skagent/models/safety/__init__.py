"""
Influence-diagram models from the AI-safety literature.

Models drawn from papers that use causal influence diagrams to state safety and
fairness properties, encoded as scikit-agent blocks and exercised through the
graphical criteria in :mod:`skagent.relevance`. Each module cites the paper its
diagrams come from and names the figures.

Kept separate from :mod:`skagent.models.benchmarks`, which collects economic
models with known solutions. Nothing here is a benchmark: what these models pin
down is graphical structure -- who observes what, who owns which payoff, what
influences what -- and their functional forms are illustrative.
"""
