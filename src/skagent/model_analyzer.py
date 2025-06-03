"""
ModelAnalyzer: Extracts structured metadata from scikit-agent DBlock/RBlock models,
preparing JSON-ready info for downstream visualization (e.g., plate-notation drawing).

Key concepts:
- instant edge: dependency within the same time period
- lag edge: dependency from previous time period (including self-lag like p_{t-1} → p_t)
- param edge: dependency from a calibration parameter
- shock edge: dependency from an exogenous shock
"""

import re
import inspect
from collections import defaultdict
from skagent.model import Control, DBlock, RBlock
from HARK.distributions import Distribution

_TOKEN_RE = re.compile(r"\b[A-Za-z_]\w*\b")


class ModelAnalyzer:
    """
    Analyze a scikit-agent DBlock or RBlock and extract:
      - node_meta: kind, agent, plate, observed for each variable
      - edges: instant / lag / param / shock dependencies (keeps self-lag)
      - formulas: human-readable equations for each variable
      - plates: loop‐notation plates inferred from agents
      - block_plates: plates assigned at the block level
    """

    def __init__(self, model, calibration, observables=None, block_agent=None):
        """
        Parameters
        ----------
        model : DBlock or RBlock
            The model to analyze
        calibration : dict
            Calibration parameters
        observables : list, optional
            Additional variables to mark as observed (for Pearl d-separation analysis)
        block_agent : str, optional
            Agent assignment at the block level. This is different from variable-level
            agent assignments. When a block is assigned to an agent/plate, all variables
            in that block (unless specifically assigned otherwise) belong to that plate.
            
            For example, in a consumption-savings model with block_agent="consumer",
            all dynamic variables (y, p, m, a, c, u) would be on the consumer plate.
        """
        self.model = model
        self.calibration = calibration.copy()
        self.observables = set(observables or [])
        self.block_agent = block_agent  # Block-level agent assignment

        # Flatten RBlock → list of DBlock(s)
        self._blocks = []
        self._walk_blocks()

        # Storage
        self.node_meta = {}
        self._raw_deps = defaultdict(list)  # target → [sources…]
        self._param_deps = defaultdict(set)  # param → {targets}
        self._time_deps = set()  # {(target, source), …} for lag edges
        self.edges = {"instant": [], "lag": [], "param": [], "shock": []}
        self.formulas = {}
        self.plates = {}
        self.block_plates = {}  # Separate tracking for block-level plates

    def _walk_blocks(self):
        """Flatten RBlock into list of DBlock(s)."""
        if isinstance(self.model, DBlock):
            self._blocks = [self.model]
        elif isinstance(self.model, RBlock):
            for blk in self.model.blocks:
                if isinstance(blk, DBlock):
                    self._blocks.append(blk)
                else:
                    raise ValueError("Only one level of RBlock supported")
        else:
            raise ValueError("Model must be a DBlock or RBlock")

    def analyze(self):
        """Run the full analysis pipeline."""
        self._collect_nodes()
        self._collect_dependencies()

        # Treat any referenced-but-unknown symbol as exogenous state
        for tgt, deps in list(self._raw_deps.items()):
            for src in deps:
                if src not in self.node_meta:
                    self.node_meta[src] = {
                        "kind": "state",
                        "agent": self.block_agent or "global",
                        "plate": self.block_agent,
                        "observed": False,
                    }

        self._identify_time_dependencies()
        self._assemble_edges()
        self._collect_formulas()
        self._collect_plates()
        return self

    def _collect_nodes(self):
        """Classify every variable and record its metadata."""
        
        for blk in self._blocks:
            # Use block_agent if specified, otherwise check if block has agent attribute
            block_plate = self.block_agent or getattr(blk, 'agent', None)
            
            # 1) Collect shocks - typically global/exogenous, not in any plate
            for var, shock_def in blk.shocks.items():
                self.node_meta[var] = {
                    "kind": "shock",
                    "agent": "global",
                    "plate": None,  # Shocks are not in any plate
                    "observed": False,
                }

            # 2) Collect dynamics - all belong to the block's plate
            for var, rule in blk.dynamics.items():
                if isinstance(rule, Control):
                    kind = "control"
                    # Control might specify its own agent for other purposes
                    agent = getattr(rule, 'agent', None) or block_plate or "global"
                else:
                    kind = "state"
                    agent = block_plate or "global"
                
                self.node_meta[var] = {
                    "kind": kind,
                    "agent": agent,
                    "plate": block_plate,  # All dynamics belong to the block's plate
                    "observed": (kind == "control") or (var in self.observables),
                }

            # 3) Collect rewards - belong to the block's plate
            for var, rule in blk.reward.items():
                self.node_meta[var] = {
                    "kind": "reward",
                    "agent": block_plate or "global",
                    "plate": block_plate,  # Rewards belong to the block's plate
                    "observed": True,  # Rewards are always observed
                }

        # 4) Collect parameters - global, not in any plate
        defined_vars = set(self.node_meta.keys())
        for p in self.calibration:
            if p not in defined_vars:
                self.node_meta[p] = {
                    "kind": "param",
                    "agent": "global",
                    "plate": None,  # Parameters are not in any plate
                    "observed": False,
                }

    def _extract_dependencies(self, rule, var_name=None):
        """
        Extract variable dependencies from different rule types.
        
        Parameters
        ----------
        rule : various
            Can be Control, Distribution, callable, or string
        var_name : str, optional
            Name of the variable (for error messages)
        
        Returns
        -------
        list
            List of dependency variable names
        """
        deps = []
        
        if isinstance(rule, Control):
            # Control has explicit information set
            deps = list(rule.iset)
        elif isinstance(rule, Distribution):
            # Distribution might depend on calibration parameters
            # This would require inspecting the distribution parameters
            # For now, we'll need to handle this case-by-case
            pass
        elif isinstance(rule, tuple) and len(rule) == 2:
            # Shock definition with (Distribution, params)
            dist_class, params = rule
            if isinstance(params, dict):
                for param_expr in params.values():
                    if isinstance(param_expr, str):
                        # Extract variables from string expressions
                        deps.extend(_TOKEN_RE.findall(param_expr))
        elif isinstance(rule, str):
            # String expression
            deps = _TOKEN_RE.findall(rule)
        elif callable(rule):
            # Callable function
            try:
                deps = list(inspect.signature(rule).parameters.keys())
            except Exception:
                # Fallback: parse source code
                try:
                    src = inspect.getsource(rule)
                    deps = _TOKEN_RE.findall(src)
                except Exception:
                    pass
        
        return deps

    def _collect_dependencies(self):
        """Collect raw dependencies and build interim param_deps."""
        for blk in self._blocks:
            # Process shocks
            for var, shock_def in blk.shocks.items():
                deps = self._extract_dependencies(shock_def, var)
                for d in deps:
                    self._raw_deps[var].append(d)
                    if d in self.calibration:
                        self._param_deps[d].add(var)

            # Process dynamics
            for var, rule in blk.dynamics.items():
                deps = self._extract_dependencies(rule, var)
                for d in deps:
                    self._raw_deps[var].append(d)
                    if d in self.calibration:
                        self._param_deps[d].add(var)

            # Process rewards
            for var, rule in blk.reward.items():
                deps = self._extract_dependencies(rule, var)
                for d in deps:
                    self._raw_deps[var].append(d)
                    if d in self.calibration:
                        self._param_deps[d].add(var)

        # Deduplicate dependencies
        for tgt in list(self._raw_deps):
            self._raw_deps[tgt] = sorted(set(self._raw_deps[tgt]))

    def _identify_time_dependencies(self):
        """
        Identify lag dependencies based on forward references in variable definitions.
        
        Key principle: If variable A depends on variable B, but B is defined AFTER A
        in the dynamics order, then A must depend on B from the previous period (B*).
        
        Example:
            dynamics = {
                "m": lambda a: a + 1,  # 'a' not yet defined, so this is a_prev
                "a": lambda m: m - 1,  # 'm' already defined, so this is m_current
            }
        
        This creates: a* -> m (lag edge) and m -> a (instant edge)
        """
        for blk in self._blocks:
            # Get ordered list of dynamic variables (order matters!)
            dynamics_vars = list(blk.dynamics.keys())
            defined_vars = set()  # Variables defined so far
            
            for var in dynamics_vars:
                # Get dependencies for this variable
                deps = self._raw_deps.get(var, [])
                
                for dep in deps:
                    # Self-dependency is always lag (e.g., p depends on p)
                    if dep == var:
                        self._time_deps.add((var, dep))
                    # If dependency is not yet defined, it's a forward reference (lag)
                    elif dep in dynamics_vars and dep not in defined_vars:
                        self._time_deps.add((var, dep))
                    # Otherwise it's an instant dependency (already handled in _assemble_edges)
                
                # Mark this variable as defined
                defined_vars.add(var)
            
            # Process rewards (they come after all dynamics)
            for var in blk.reward.keys():
                deps = self._raw_deps.get(var, [])
                for dep in deps:
                    # Rewards can only have instant dependencies since all dynamics are defined
                    pass  # No time dependencies for rewards

    def _assemble_edges(self):
        """
        Convert raw_deps + time_deps + param_deps into four classified edge lists.
        Preserves self-lag edges (p→p) for visualization splitting.
        """
        for tgt, deps in self._raw_deps.items():
            for src in deps:
                # Skip if source is the same as target AND it's not a lag dependency
                if src == tgt and (tgt, src) not in self._time_deps:
                    continue
                
                # 1) Lag edges (including self-lag)
                if (tgt, src) in self._time_deps:
                    self.edges["lag"].append((src, tgt))
                # 2) Parameter edges
                elif src in self._param_deps:
                    self.edges["param"].append((src, tgt))
                # 3) Shock edges
                elif src in self.node_meta and self.node_meta[src]["kind"] == "shock":
                    self.edges["shock"].append((src, tgt))
                # 4) Instant edges
                else:
                    self.edges["instant"].append((src, tgt))

        # Deduplicate and sort
        for edge_type in self.edges:
            self.edges[edge_type] = sorted(set(self.edges[edge_type]))

    def _collect_formulas(self):
        """Generate human-readable formulas for each variable."""
        for blk in self._blocks:
            # Process dynamics
            for var, rule in blk.dynamics.items():
                if isinstance(rule, Control):
                    deps = sorted(rule.iset)
                    bounds_info = []
                    if rule.lower_bound:
                        bounds_info.append("lower_bound")
                    if rule.upper_bound:
                        bounds_info.append("upper_bound")
                    bounds_str = f", {', '.join(bounds_info)}" if bounds_info else ""
                    self.formulas[var] = f"{var} = Control({', '.join(deps)}{bounds_str})"
                elif isinstance(rule, str):
                    self.formulas[var] = f"{var} = {rule}"
                else:
                    self._format_callable_formula(var, rule)

            # Process rewards
            for var, rule in blk.reward.items():
                self._format_callable_formula(var, rule)

        # Process parameters
        for p, val in self.calibration.items():
            if p in self.node_meta and self.node_meta[p]["kind"] == "param":
                self.formulas[p] = f"{p} = {val}"

    def _format_callable_formula(self, var, rule):
        """Helper to format callable rules into formulas."""
        try:
            src = inspect.getsource(rule).strip()
            if "lambda" in src:
                # Extract lambda body
                body = src.split(":", 1)[1].strip()
                # Remove trailing comma or parenthesis
                body = body.rstrip(",)")
                self.formulas[var] = f"{var} = {body}"
            else:
                self.formulas[var] = f"{var} = [Function]"
        except Exception:
            self.formulas[var] = f"{var} = [Unknown]"

    def _collect_plates(self):
        """Build plates mapping from blocks that have agents."""
        # Collect all unique plates from node metadata
        plates_set = set()
        for meta in self.node_meta.values():
            if meta["plate"]:
                plates_set.add(meta["plate"])
        
        # Create plate information
        for plate_name in plates_set:
            self.plates[plate_name] = {
                "label": plate_name.capitalize(),
                "size": f"N_{plate_name}"
            }

    def to_dict(self):
        """Return a JSON-serializable dict of the analysis."""
        return {
            "node_meta": self.node_meta,
            "edges": self.edges,
            "formulas": self.formulas,
            "plates": self.plates,
        }