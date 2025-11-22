"""
ModelAnalyzer: Extracts structured metadata from scikit-agent DBlock/RBlock models
for visualization (e.g., plate-notation drawing).

Key concepts:
- instant edge: dependency within the same time period
- lag edge: dependency from previous time period (including self-lag like p_{t-1} → p_t)
- param edge: dependency from a calibration parameter
- shock edge: dependency from an exogenous shock
"""

from collections import defaultdict
from skagent.rule import extract_dependencies


class ModelAnalyzer:
    """
    Analyze a scikit-agent DBlock or RBlock and extract:
      - node_meta: kind, agent, plate, observed for each variable
      - edges: instant / lag / param / shock dependencies
      - plates: loop-notation plates inferred from agents
    """

    def __init__(self, model, calibration, block_agent=None):
        """
        Parameters
        ----------
        model : DBlock or RBlock
            The model to analyze
        calibration : dict
            Calibration parameters
        block_agent : str, optional
            Agent/plate assignment at the block level
        """
        self.model = model
        self.calibration = calibration
        self.block_agent = block_agent

        # Storage
        self.node_meta = {}
        self.edges = {"instant": [], "lag": [], "param": [], "shock": []}
        self.plates = {}

        # Internal state for analysis
        self._blocks = list(self.model.iter_dblocks())
        self._raw_deps = defaultdict(list)
        self._time_deps = set()

    def analyze(self):
        """Run the full analysis pipeline."""
        self._collect_nodes()
        self._collect_dependencies()
        self._identify_time_dependencies()
        self._assemble_edges()
        self._collect_plates()
        self._add_lag_variables()
        return self

    def _collect_nodes(self):
        """Classify every variable and record its metadata."""
        from skagent.block import Control  # TODO: move to separate module

        for blk in self._blocks:
            plate = self.block_agent or getattr(blk, "agent", None)

            # Shocks - no plate assignment
            for var in blk.get_shocks():
                self.node_meta[var] = {
                    "kind": "shock",
                    "agent": "global",
                    "plate": None,
                    "observed": False,
                }

            # Dynamics - assigned to block's plate
            for var, rule in blk.get_dynamics().items():
                if isinstance(rule, Control):
                    kind = "control"
                    agent = rule.agent or plate or "global"
                    if not isinstance(agent, str):
                        agent = str(agent) if agent else "global"
                    observed = True
                else:
                    kind = "state"
                    agent = plate or "global"
                    observed = False

                self.node_meta[var] = {
                    "kind": kind,
                    "agent": agent,
                    "plate": plate,
                    "observed": observed,
                }

            # Rewards - use the agent assignment from reward dictionary
            for var, agent_name in blk.reward.items():
                if not isinstance(agent_name, str):
                    agent_name = str(agent_name) if agent_name else "global"

                self.node_meta[var] = {
                    "kind": "reward",
                    "agent": agent_name,
                    "plate": plate,
                    "observed": True,
                }

        # Parameters - no plate assignment
        for param in self.calibration:
            if param not in self.node_meta:
                self.node_meta[param] = {
                    "kind": "param",
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

    def _identify_time_dependencies(self):
        """
        Use get_arrival_states method to identify lag dependencies.
        """
        for blk in self._blocks:
            arrival_states = blk.get_arrival_states(self.calibration)
            for var in blk.get_dynamics().keys():
                for dep in self._raw_deps.get(var, []):
                    if dep in arrival_states:
                        self._time_deps.add((var, dep))

    def _assemble_edges(self):
        """Convert dependencies into classified edge lists."""
        for target, deps in self._raw_deps.items():
            for source in deps:
                if source == target and (target, source) not in self._time_deps:
                    continue

                if (target, source) in self._time_deps:
                    self.edges["lag"].append((source, target))
                elif source in self.calibration:
                    self.edges["param"].append((source, target))
                elif self.node_meta.get(source, {}).get("kind") == "shock":
                    self.edges["shock"].append((source, target))
                else:
                    self.edges["instant"].append((source, target))

        for edge_type in self.edges:
            self.edges[edge_type] = sorted(set(self.edges[edge_type]))

    def _collect_plates(self):
        """
        Build plates from unique plate assignments at the block level,
        AND from agent assignments at the variable level.
        """
        # 1. Collect plates explicitly defined at the block level
        plates_from_blocks = {
            meta["plate"] for meta in self.node_meta.values() if meta["plate"]
        }

        # 2. Collect agents assigned to specific variables (and treat them as plates)
        # This correctly finds agents defined on Control objects.
        plates_from_agents = {
            meta["agent"]
            for meta in self.node_meta.values()
            if meta.get("agent") and meta["agent"] != "global"
        }

        # 3. Combine both sources to get a complete set of plates
        plates_set = plates_from_blocks.union(plates_from_agents)

        for plate_name in plates_set:
            self.plates[plate_name] = {
                "label": plate_name.capitalize(),
                "size": "",
            }

    def _add_lag_variables(self):
        """Add metadata for lag variables (e.g., p* for p_{t-1})."""
        lag_sources = {source for _, source in self._time_deps}

        for source in lag_sources:
            lag_var = f"{source}*"
            if source in self.node_meta and lag_var not in self.node_meta:
                self.node_meta[lag_var] = self.node_meta[source].copy()
                self.node_meta[lag_var]["observed"] = False

    def to_dict(self):
        """Return a JSON-serializable dict of the analysis."""
        return {
            "node_meta": self.node_meta,
            "edges": self.edges,
            "plates": self.plates,
        }
