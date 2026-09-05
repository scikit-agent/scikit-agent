"""
ModelAnalyzer: Extracts structured metadata from scikit-agent DBlock/RBlock models.

The analyzer builds an annotated dependency graph (``self.G``) as its source of
truth and exposes it in two forms:

- ``to_dict()`` -- node metadata and classified edges for visualization
  (e.g., plate-notation drawing via ModelVisualizer).
- ``influence_graph()`` -- the influence-diagram (SCIM) view consumed by
  :mod:`skagent.relevance` for strategic-relevance analysis.

Key concepts:
- instant edge: dependency within the same time period
- lag edge: dependency from previous time period (including self-lag like p_{t-1} → p_t)
- param edge: dependency from a calibration parameter
- shock edge: dependency from an exogenous shock
"""

from collections import defaultdict

import networkx as nx

from skagent.influence import SCIM
from skagent.rule import extract_dependencies


class ModelAnalyzer:
    """
    Analyze a scikit-agent DBlock or RBlock and extract:
      - node_meta: kind, agent, plate, observed for each variable
      - edges: instant / lag / param / shock dependencies
      - plates: the entity classes the block tree declares
    """

    def __init__(self, model, calibration, block_agent=None, discount=None):
        """
        Parameters
        ----------
        model : DBlock or RBlock
            The model to analyze
        calibration : dict
            Calibration parameters
        block_agent : str, optional
            Agent/plate assignment at the block level
        discount : str, optional
            The calibration symbol serving as the discount factor. It is
            classified ``discount`` rather than ``param``, which is what marks
            it as a variable the model was solved against rather than one its
            equations read. A block does not know this on its own, since the
            choice lives on the period built over it.
        """
        self.model = model
        self.calibration = calibration
        self.block_agent = block_agent
        self.discount = discount

        # Storage
        self.G = nx.DiGraph()  # annotated dependency graph: the source of truth
        self.node_meta = {}
        self.edges = {"instant": [], "lag": [], "param": [], "shock": []}
        self.plates = {}

        # Internal state for analysis
        self._blocks = list(self.model.iter_dblocks())
        self._raw_deps = defaultdict(list)
        self._time_deps = set()

    def analyze(self):
        """Run the full analysis pipeline.

        The annotated dependency graph ``self.G`` is the source of truth; the
        public ``node_meta`` / ``edges`` / ``plates`` are derived from it.
        """
        self._collect_nodes()
        self._collect_dependencies()
        self._identify_time_dependencies()
        self._build_graph()
        self._derive_node_meta_and_edges()
        self._collect_plates()
        self._add_lag_variables()
        return self

    def _collect_nodes(self):
        """Classify every variable and record its metadata."""
        from skagent.block import Control  # TODO: move to separate module

        signatures = self.model.signatures()

        def plate_of(var):
            """The entity class *var* is an attribute of, if exactly one is.

            A symbol declared inside an entity class is drawn inside that
            class's plate. A symbol of no class, or of several nested ones
            (which nothing yet builds, and which has no innermost class to
            pick), is drawn outside every plate.
            """
            classes = signatures.get(var, frozenset())
            if len(classes) == 1:
                return next(iter(classes))
            return self.block_agent

        for blk in self._blocks:
            # A control that declares no agent, in a block whose rewards all
            # belong to one agent, belongs to that agent: there is no other
            # candidate. With several reward owners the control must say which.
            reward_agents = {str(a) for a in blk.reward.values() if a}
            sole_reward_agent = (
                next(iter(reward_agents)) if len(reward_agents) == 1 else None
            )

            # Shocks - plated when the class declares them per instance
            for var in blk.get_shocks():
                self.node_meta[var] = {
                    "kind": "shock",
                    "agent": "global",
                    "plate": plate_of(var),
                    "observed": False,
                }

            # Dynamics - plated by the entity class they are an attribute of
            for var, rule in blk.get_dynamics().items():
                if isinstance(rule, Control):
                    kind = "control"
                    agent = (
                        rule.agent or self.block_agent or sole_reward_agent or "global"
                    )
                    if not isinstance(agent, str):
                        agent = str(agent) if agent else "global"
                    observed = True
                else:
                    kind = "state"
                    agent = self.block_agent or "global"
                    observed = False

                self.node_meta[var] = {
                    "kind": kind,
                    "agent": agent,
                    "plate": plate_of(var),
                    "observed": observed,
                }

            # Rewards - use the agent assignment from reward dictionary
            for var, agent_name in blk.reward.items():
                if not isinstance(agent_name, str):
                    agent_name = str(agent_name) if agent_name else "global"

                self.node_meta[var] = {
                    "kind": "reward",
                    "agent": agent_name,
                    "plate": plate_of(var),
                    "observed": True,
                }

        # Parameters - axis-free, so never plated
        for param in self.calibration:
            if param not in self.node_meta:
                self.node_meta[param] = {
                    "kind": "discount" if param == self.discount else "param",
                    "agent": "global",
                    "plate": None,
                    "observed": False,
                }

    def _collect_dependencies(self):
        """Extract dependencies using rule module."""
        for blk in self._blocks:
            all_rules = {**blk.get_shocks(), **blk.get_dynamics()}

            for var, rule in all_rules.items():
                deps = extract_dependencies(rule)
                self._raw_deps[var] = deps

                for dep in deps:
                    if dep not in self.node_meta:
                        self.node_meta[dep] = {
                            "kind": "state",
                            "agent": self.block_agent or "global",
                            "plate": self.block_agent,
                            "observed": False,
                        }

    def _dynamics_positions(self):
        """Position of each symbol's assignment in declaration order.

        Concatenated across blocks in execution order. A symbol assigned more
        than once keeps its first position.
        """
        position = {}
        for blk in self._blocks:
            for sym in blk.get_dynamics():
                position.setdefault(sym, len(position))
        return position

    def _identify_time_dependencies(self):
        """Identify dependencies that read a lagged (arrival) value.

        Dynamics run in declaration order, so a dependency reads its symbol's
        pre-assignment value unless that symbol is assigned earlier in the
        order. A self-reference is such a case, as is a dependency on a symbol
        the block never assigns.

        A pre-assignment value arrives from the previous period, which makes the
        dependency lagged -- unless the calibration supplies a value for the
        symbol, or the symbol is a shock realized within the period. Neither of
        those reaches back a period, so neither is lagged.

        A symbol the model ASSIGNS is not a parameter, whatever the calibration
        holds for it: the calibration is supplying its arrival value for the
        first period. So the calibration escape applies only to symbols no block
        assigns. Reading one that is assigned later in the period -- as the
        normalized consumption block's ``b`` reads the ``R`` that the portfolio
        block goes on to compute -- is a read of last period's value.
        """
        position = self._dynamics_positions()
        shocks = {s for blk in self._blocks for s in blk.get_shocks()}
        for blk in self._blocks:
            for var in blk.get_dynamics():
                if var not in position:
                    continue
                for dep in self._raw_deps.get(var, []):
                    if dep in position:
                        if position[dep] < position[var]:
                            continue  # the value assigned earlier this period
                    elif dep in self.calibration or dep in shocks:
                        continue
                    self._time_deps.add((var, dep))

    def _classify_edge(self, source, target):
        """Return the edge kind for a ``source -> target`` dependency."""
        if (target, source) in self._time_deps:
            return "lag"
        if source in self.calibration:
            return "param"
        if self.node_meta.get(source, {}).get("kind") == "shock":
            return "shock"
        return "instant"

    def _build_graph(self):
        """Build the annotated dependency graph ``self.G`` (the source of truth).

        Nodes carry ``kind`` / ``agent`` / ``plate`` / ``observed``; each edge
        carries a ``kind`` attribute (instant / lag / param / shock), replacing
        the former four parallel edge lists.
        """
        for var, meta in self.node_meta.items():
            self.G.add_node(var, **meta)

        for target, deps in self._raw_deps.items():
            for source in deps:
                if source == target and (target, source) not in self._time_deps:
                    continue
                self.G.add_edge(
                    source, target, kind=self._classify_edge(source, target)
                )

    def _derive_node_meta_and_edges(self):
        """Derive the public ``node_meta`` and classified ``edges`` from ``self.G``."""
        self.node_meta = {n: dict(self.G.nodes[n]) for n in self.G.nodes}

        edges = {"instant": [], "lag": [], "param": [], "shock": []}
        for source, target, data in self.G.edges(data=True):
            edges[data["kind"]].append((source, target))
        self.edges = {kind: sorted(set(pairs)) for kind, pairs in edges.items()}

    def _collect_plates(self):
        """Build the plates: the model's declared entity classes.

        A plate is a class the model says it has several of, so it is read from
        the block tree's entity declarations and sized from the calibration.
        An agent role is not a plate: one agent may hold several decisions and
        several agents may be instances of one class, so drawing a box per role
        boxes the wrong thing.
        """
        entities = self.model.entities()

        for name in entities:
            self.plates[name] = {
                "label": name,
                "size": self.calibration.get(name, ""),
            }

        # A plate assigned to a node by *block_agent* is drawn as well, so that
        # the caller-supplied grouping does not silently vanish.
        assigned = {meta["plate"] for meta in self.node_meta.values() if meta["plate"]}
        for name in assigned - set(entities):
            self.plates[name] = {"label": name, "size": ""}

    def _add_lag_variables(self):
        """Add metadata for lag variables (e.g., p* for p_{t-1})."""
        lag_sources = {source for _, source in self._time_deps}

        for source in lag_sources:
            lag_var = f"{source}*"
            if source in self.node_meta and lag_var not in self.node_meta:
                self.node_meta[lag_var] = self.node_meta[source].copy()
                self.node_meta[lag_var]["observed"] = False

    def influence_graph(self, dynamic=False):
        """Return the SCIM (influence-diagram) view for strategic-relevance analysis.

        The graph :mod:`skagent.relevance` consumes: chance / decision / utility
        nodes with the causal (instant + shock) edges between them. Parameter
        nodes are dropped -- they are deterministic constants, not random
        variables, and leaving them in would open spurious d-connection paths
        (an un-conditioned fork ``A <- p -> B``) that corrupt s-reachability.
        Lag edges are excluded here (single-period scope); cross-period reliance
        is handled by the unrolling machinery separately.

        Parameters
        ----------
        dynamic : bool, optional
            Make the diagram faithful to one period of a recurring problem, by
            splitting each reassigned variable's arrival value into its own
            ``<name>*`` node and adding a continuation-value utility node per
            deciding agent
            (:meth:`skagent.influence.SCIM.with_lagged_arrivals`,
            :meth:`skagent.influence.SCIM.with_continuation`). Without this a
            single-period projection is blind to payoffs arriving through the
            next period's value, and conflates a variable's arrival value with
            the value it is reassigned to. With it, a decision's parents are its
            information set. Off by default, since it changes the node set
            existing callers see.

        Returns
        -------
        skagent.influence.SCIM
        """
        kind_map = {
            "shock": "chance",
            "state": "chance",
            "control": "decision",
            "reward": "utility",
        }

        scim = nx.DiGraph()
        for node in self.G.nodes:
            attrs = self.G.nodes[node]
            scim_kind = kind_map.get(attrs["kind"])
            if scim_kind is None:  # drop parameter nodes
                continue
            scim.add_node(node, kind=scim_kind, agent=attrs["agent"])

        for source, target, data in self.G.edges(data=True):
            if (
                data["kind"] in ("instant", "shock")
                and source in scim
                and target in scim
            ):
                scim.add_edge(source, target)

        decisions = [n for n in scim.nodes if scim.nodes[n]["kind"] == "decision"]
        decision_agent = {d: scim.nodes[d]["agent"] for d in decisions}
        agent_utilities = defaultdict(list)
        for node in scim.nodes:
            if scim.nodes[node]["kind"] == "utility":
                agent_utilities[scim.nodes[node]["agent"]].append(node)
        agent_utilities = dict(agent_utilities)

        view = SCIM(scim, decisions, agent_utilities, decision_agent)

        if dynamic:
            # Lag edges were dropped above; reintroduce them as arrival-value
            # nodes so a decision's parents are its information set.
            view = view.with_lagged_arrivals(self._time_deps).with_continuation(
                self.model.get_arrival_states(self.calibration)
            )

        return view

    def to_dict(self):
        """Return a JSON-serializable dict of the analysis."""
        return {
            "node_meta": self.node_meta,
            "edges": self.edges,
            "plates": self.plates,
        }
