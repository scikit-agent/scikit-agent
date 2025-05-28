"""
ModelAnalyzer: Extracts structured metadata from scikit-agent DBlock/RBlock models,
preparing JSON-ready info for downstream visualization (e.g., plate-notation drawing).
"""

import re
import inspect
from collections import defaultdict
from skagent.model import Control, DBlock, RBlock

_TOKEN_RE = re.compile(r"\b[A-Za-z_]\w*\b")


class ModelAnalyzer:
    """
    Analyze a scikit-agent DBlock or RBlock and extract:
      - node_meta: kind, agent, plate, observed for each variable
      - edges: instant / lag / param / shock dependencies (keeps self-lag)
      - formulas: human-readable equations for each variable
      - plates: loop‐notation plates inferred from agents
    """

    def __init__(self, model, calibration, observables=None):
        self.model = model
        self.calibration = calibration.copy()
        self.observables = set(observables or [])

        # Flatten RBlock → list of DBlock(s)
        self._blocks = []
        self._walk_blocks()

        # Storage
        self.node_meta = {}
        self._raw_deps = defaultdict(list)  # target → [sources…]
        self._param_deps = defaultdict(set)  # param → {targets}
        self._prev_deps = set()  # {(target, source), …}
        self.edges = {"instant": [], "lag": [], "param": [], "shock": []}
        self.formulas = {}
        self.plates = {}

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
                        "agent": "global",
                        "plate": None,
                        "observed": False,
                    }

        self._identify_time_dependencies()
        self._assemble_edges()
        self._collect_formulas()
        self._collect_plates()
        return self

    def _collect_nodes(self):
        """Classify every variable and record its metadata."""
        # 1) all LHS names in dynamics & reward
        lhs = set()
        for blk in self._blocks:
            lhs |= set(blk.dynamics.keys())
            lhs |= set(blk.reward.keys())

        # 2) shocks
        for blk in self._blocks:
            for var, shock_def in blk.shocks.items():
                agent = getattr(shock_def, "agent", "global") or "global"
                self.node_meta[var] = {
                    "kind": "shock",
                    "agent": agent,
                    "plate": agent if agent != "global" else None,
                    "observed": False,
                }

        # 3) dynamics
        for blk in self._blocks:
            for var, rule in blk.dynamics.items():
                kind = "control" if isinstance(rule, Control) else "state"
                agent = getattr(rule, "agent", "global") or "global"
                self.node_meta[var] = {
                    "kind": kind,
                    "agent": agent,
                    "plate": agent if agent != "global" else None,
                    "observed": (kind in ("control", "reward"))
                    or (var in self.observables),
                }

        # 4) reward
        for blk in self._blocks:
            for var, rd in blk.reward.items():
                agent = (
                    rd
                    if isinstance(rd, str)
                    else getattr(rd, "agent", "global") or "global"
                )
                self.node_meta[var] = {
                    "kind": "reward",
                    "agent": agent,
                    "plate": agent if agent != "global" else None,
                    "observed": True,
                }

        # 5) params: only keys in calibration not appearing on any LHS
        for p in set(self.calibration) - lhs:
            self.node_meta.setdefault(
                p,
                {"kind": "param", "agent": "global", "plate": None, "observed": False},
            )

    def _rhs_symbols(self, rule):
        """Extract variable names from RHS rule (callable or string)."""
        if isinstance(rule, str):
            return _TOKEN_RE.findall(rule)
        if callable(rule):
            try:
                return list(inspect.signature(rule).parameters.keys())
            except Exception:
                src = inspect.getsource(rule)
                return _TOKEN_RE.findall(src)
        return []

    def _collect_dependencies(self):
        """Collect raw dependencies and build interim param_deps."""
        for blk in self._blocks:
            # shock parameters → shock var
            for var, sd in blk.shocks.items():
                if isinstance(sd, tuple) and len(sd) == 2:
                    _, params = sd
                    if isinstance(params, dict):
                        for v in params.values():
                            if isinstance(v, str):
                                self._raw_deps[var].append(v)
                                self._param_deps[v].add(var)

            # dynamics
            for var, rule in blk.dynamics.items():
                deps = (
                    list(rule.iset)
                    if isinstance(rule, Control)
                    else self._rhs_symbols(rule)
                )
                for d in deps:
                    self._raw_deps[var].append(d)
                    if d in self.calibration:
                        self._param_deps[d].add(var)

            # reward
            for var, rd in blk.reward.items():
                deps = self._rhs_symbols(rd) if callable(rd) else []
                for d in deps:
                    self._raw_deps[var].append(d)
                    if d in self.calibration:
                        self._param_deps[d].add(var)

        # dedupe
        for tgt in list(self._raw_deps):
            self._raw_deps[tgt] = sorted(set(self._raw_deps[tgt]))

    def _identify_time_dependencies(self):
        """Mark (tgt, src) pairs for lag edges via insertion-order heuristic."""
        for blk in self._blocks:
            order = list(blk.dynamics.keys())
            seen = set()
            for tgt in order:
                deps = [d for d in self._raw_deps.get(tgt, []) if d in order]
                for src in deps:
                    if src == tgt or src not in seen:
                        self._prev_deps.add((tgt, src))
                seen.add(tgt)

    def _assemble_edges(self):
        """
        Convert raw_deps + prev_deps + param_deps into four classified edge lists.
        Preserve self-lag edges (p→p) for visualization splitting.
        """
        for tgt, deps in self._raw_deps.items():
            for src in deps:
                # 1) lag edges, including self-lag
                if (tgt, src) in self._prev_deps:
                    self.edges["lag"].append((src, tgt))
                    continue
                # 2) drop any remaining self-loop
                if src == tgt:
                    continue
                # 3) param edges
                if src in self._param_deps:
                    self.edges["param"].append((src, tgt))
                # 4) shock edges
                elif src in self.node_meta and self.node_meta[src]["kind"] == "shock":
                    self.edges["shock"].append((src, tgt))
                # 5) instant edges
                else:
                    self.edges["instant"].append((src, tgt))

        # dedupe & sort
        for et in self.edges:
            self.edges[et] = sorted(set(self.edges[et]))

    def _collect_formulas(self):
        """Generate human-readable formulas for each variable."""
        for blk in self._blocks:
            # dynamics
            for var, rule in blk.dynamics.items():
                if isinstance(rule, Control):
                    deps = sorted(rule.iset)
                    self.formulas[var] = f"{var} = Control({', '.join(deps)})"
                elif isinstance(rule, str):
                    self.formulas[var] = f"{var} = {rule}"
                else:
                    try:
                        src = inspect.getsource(rule).strip()
                        if "lambda" in src:
                            body = src.split(":", 1)[1].strip().rstrip(",")
                            self.formulas[var] = f"{var} = {body}"
                        else:
                            self.formulas[var] = f"{var} = [Function]"
                    except Exception:
                        self.formulas[var] = f"{var} = [Unknown]"

            # reward
            for var, rd in blk.reward.items():
                if callable(rd):
                    try:
                        src = inspect.getsource(rd).strip()
                        if "lambda" in src:
                            body = src.split(":", 1)[1].strip().rstrip(",")
                        else:
                            body = "[Function]"
                        self.formulas[var] = f"{var} = {body}"
                    except Exception:
                        self.formulas[var] = f"{var} = [Unknown]"

        # params
        for p, val in self.calibration.items():
            if p in self.node_meta and self.node_meta[p]["kind"] == "param":
                self.formulas[p] = f"{p} = {val}"

    def _collect_plates(self):
        """Build plates mapping from non-global agents."""
        for var, meta in self.node_meta.items():
            agent = meta["agent"]
            if agent != "global":
                self.plates.setdefault(
                    agent, {"label": agent.capitalize(), "size": f"N_{agent}"}
                )

    def to_dict(self):
        """Return a JSON-serializable dict of the analysis."""
        return {
            "node_meta": self.node_meta,
            "edges": self.edges,
            "formulas": self.formulas,
            "plates": self.plates,
        }
