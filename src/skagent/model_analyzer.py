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
from skagent.model import Control, DBlock, RBlock
from skagent.distributions import Distribution
from skagent.rule import format_rule, extract_dependencies


class ModelAnalyzer:
    """
    Analyze a scikit-agent DBlock or RBlock and extract:
      - node_meta: kind, agent, plate, observed for each variable
      - edges: instant / lag / param / shock dependencies
      - formulas: human-readable equations for each variable
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
        self.formulas = {}
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
        self._collect_formulas()
        self._collect_plates()
        self._add_lag_variables()
        return self

    def _collect_nodes(self):
        """Classify every variable and record its metadata."""
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
                    # Ensure agent is a string (defensive programming)
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
                # Ensure agent_name is a string (defensive programming)
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
            # Collect from dynamics and shocks
            all_rules = {**blk.get_shocks(), **blk.get_dynamics()}
            
            for var, rule in all_rules.items():
                deps = extract_dependencies(rule)
                self._raw_deps[var] = deps
                
                # Handle unknown dependencies as external states
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
            # Use the official method to get arrival states
            arrival_states = blk.get_arrival_states(self.calibration)
            
            # Any dependency on an arrival state is a lag dependency
            for var in blk.get_dynamics().keys():
                for dep in self._raw_deps.get(var, []):
                    if dep in arrival_states:
                        self._time_deps.add((var, dep))

    def _assemble_edges(self):
        """Convert dependencies into classified edge lists."""
        for target, deps in self._raw_deps.items():
            for source in deps:
                # Skip self-loops unless they're lag dependencies
                if source == target and (target, source) not in self._time_deps:
                    continue
                
                # Classify edge type
                if (target, source) in self._time_deps:
                    self.edges["lag"].append((source, target))
                elif source in self.calibration:
                    self.edges["param"].append((source, target))
                elif self.node_meta.get(source, {}).get("kind") == "shock":
                    self.edges["shock"].append((source, target))
                else:
                    self.edges["instant"].append((source, target))
        
        # Deduplicate
        for edge_type in self.edges:
            self.edges[edge_type] = sorted(set(self.edges[edge_type]))

    def _collect_formulas(self):
        """Generate formulas using rule module."""
        for blk in self._blocks:
            # Process dynamics only - rewards don't contain rules, just agent assignments
            for var, rule in blk.get_dynamics().items():
                self.formulas[var] = format_rule(var, rule)
        
        # Process parameters
        for param, value in self.calibration.items():
            if param in self.node_meta:
                self.formulas[param] = format_rule(param, value)

    def _collect_plates(self):
        """Build plates from unique plate assignments."""
        plates_set = {meta["plate"] for meta in self.node_meta.values() if meta["plate"]}
        
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
            "formulas": self.formulas,
            "plates": self.plates,
        }