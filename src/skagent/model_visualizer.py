import os
import yaml
import pydot
import colorsys
import random

# 1) Load config
_CONFIG_FILE = os.path.join(
    os.path.dirname(__file__), "model_visualization_config.yaml"
)
with open(_CONFIG_FILE, "r") as f:
    _CFG = yaml.safe_load(f)


class ModelVisualizer:
    def __init__(self, analysis):
        """
        analysis = {
          node_meta, edges, plates, formulas
        }
        """
        self.meta = analysis["node_meta"]
        self.edges = analysis["edges"]
        self.plates = analysis["plates"]
        self.nodes = {}
        # aliases for convenience
        self.vs = _CFG["variable_shapes"]
        self.ns = _CFG["node_styles"]
        self.es = _CFG["edge_styles"]
        self.cc = _CFG["color_config"]
        self.cg = _CFG["color_generation"]
        self.gl = _CFG["graph_layout"]
        self.gc = _CFG["graph_config"]

        # prepare agent→color mapping
        self.agent_colors = self._build_agent_colors()

    def _build_agent_colors(self):
        """
        Golden‐ratio HSL generator for each agent (except 'global'/'other'),
        global/other use default_other_color.
        """
        agents = sorted({m["agent"] for m in self.meta.values()})
        # seed
        seed = self.cg.get("color_seed", 0)
        random.seed(seed)
        # pick initial hue random in [0,1)
        h0 = random.random()
        phi = self.cg.get("golden_ratio_factor", 0.618033988749895)
        sat_lo, sat_hi = self.cg["saturation_range"]
        light_lo, light_hi = self.cg["lightness_range"]

        mapping = {}
        idx = 0
        for agent in agents:
            if agent in ("global", "other"):
                mapping[agent] = self.cc["default_other_color"]
            else:
                h = (h0 + idx * phi) % 1.0
                s = (sat_lo + sat_hi) / 2
                lightness = (light_lo + light_hi) / 2
                # HSL → RGB → hex
                r, g, b = colorsys.hls_to_rgb(h, lightness, s)
                mapping[agent] = "#{:02x}{:02x}{:02x}".format(
                    int(r * 255), int(g * 255), int(b * 255)
                )
                idx += 1
        return mapping

    def _node_shape(self, kind):
        """Pick shape by variable kind."""
        key = {
            "shock": "shock_vars",
            "state": "state_vars",
            "control": "control_vars",
            "reward": "reward_vars",
            "param": "param_vars",
        }.get(kind, None) or "default"
        return self.vs.get(key, self.vs["default"])

    def _make_node(self, name):
        """Create or return a styled pydot.Node."""
        if name in self.nodes:
            return self.nodes[name]
        m = self.meta.get(name, {})
        kind = m.get("kind", "state")
        agent = m.get("agent", "other")
        observed = m.get("observed", False)

        style = dict(self.ns["default"])  # base
        # add shape
        style["shape"] = self._node_shape(kind)
        # fill by agent color
        fill = self.agent_colors.get(agent, self.cc["default_other_color"])
        style["style"] = style.get("style", "") + ",filled"
        style["fillcolor"] = fill
        # observed override
        if observed:
            style.update(self.ns.get("previous_period", {}))

        # font etc from default
        node = pydot.Node(name, **style)
        self.nodes[name] = node
        return node

    def create_graph(self):
        # 1) New graph
        gconf = self.gc
        graph = pydot.Dot(graph_type=gconf["graph_type"])
        # title
        graph.set_label(gconf["title"])
        graph.set_labelloc("t")
        # layout
        graph.set("rankdir", self.gl["rankdir"])
        graph.set("nodesep", str(self.gl["node_padding"]))

        # 2) pre-create nodes (incl. ALL prev-period nodes for lag edges)
        for var in self.meta:
            self._make_node(var)

        # Create previous period nodes for ALL lag edges
        for src, tgt in self.edges.get("lag", []):
            prev = f"{src}*"
            # Create metadata for previous period node if not exists
            if prev not in self.meta:
                # Inherit properties from current period node
                src_meta = self.meta.get(src, {})
                self.meta[prev] = {
                    "kind": src_meta.get("kind", "state"),
                    "agent": src_meta.get("agent", "other"),
                    "observed": False,
                    "plate": None,  # Previous period nodes are NOT in any plate
                }
            self._make_node(prev)

        # 3) plates = subgraphs
        plate_subs = {}
        for agent, info in self.plates.items():
            lbl = (
                info["label"].capitalize()
                if self.gl["cluster_label_capitalize"]
                else info["label"]
            )
            sg = pydot.Cluster(
                graph_name=f"cluster_{agent}",
                label=f"{info['size']} {lbl}",
                labeljust="r",
                style=self.gl["cluster_style"],
                fillcolor=self.gl["cluster_fillcolor"],
                fontsize=str(self.gl["plate_fontsize"]),
            )
            graph.add_subgraph(sg)
            plate_subs[agent] = sg

        # 4) add nodes to plates or root
        for name, m in self.meta.items():
            # Skip nodes that are already added as part of lag handling
            if name.endswith("*"):
                continue

            node = self.nodes[name]
            pl = m.get("plate")
            if pl and pl in plate_subs:
                plate_subs[pl].add_node(node)
            else:
                graph.add_node(node)

        # Add ALL previous period nodes to root graph (outside any plate)
        for src, tgt in self.edges.get("lag", []):
            prev = f"{src}*"
            if prev in self.nodes:
                graph.add_node(self.nodes[prev])

        # 5) edges
        def add_edge(src, tgt, etype):
            style = dict(self.es.get(etype, self.es["default"]))
            e = pydot.Edge(self.nodes[src], self.nodes[tgt], **style)
            graph.add_edge(e)

        # instant & param & shock as "current_period"
        for et in ("instant", "param", "shock"):
            for src, tgt in self.edges.get(et, []):
                add_edge(src, tgt, "current_period")

        # lag edges - ALWAYS use previous period nodes
        for src, tgt in self.edges.get("lag", []):
            prev = f"{src}*"
            add_edge(prev, tgt, "previous_period")

        return graph
