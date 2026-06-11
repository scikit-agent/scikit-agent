import os
import yaml
import pydot
import colorsys
import random
import matplotlib.image as mpimg
from io import BytesIO
from IPython.display import SVG, display


# 1) Load config
_CONFIG_FILE = os.path.join(
    os.path.dirname(__file__), "model_visualization_config.yaml"
)
with open(_CONFIG_FILE, "r") as f:
    _CFG = yaml.safe_load(f)


class ModelVisualizer:
    def __init__(self, analysis, title=None):
        """
        analysis = {
          node_meta, edges, plates, formulas
        }
        """
        self.title = title
        self.meta = analysis["node_meta"]
        self.edges = analysis["edges"]
        self.plates = analysis["plates"]
        self.formulas = analysis.get("formulas", {})
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
        Golden-ratio HSL generator for each agent (except 'global'/'other'),
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
        # MODIFIED: Directly use 'kind' to look up the shape, falling back to default.
        return self.vs.get(kind, self.vs["default"])

    def _make_node(self, name):
        """Create or return a styled pydot.Node."""
        if name in self.nodes:
            return self.nodes[name]
        m = self.meta.get(name, {})
        kind = m.get("kind", "state")
        agent = m.get("agent", "other")

        style = dict(self.ns["default"])  # base
        # add shape
        style["shape"] = self._node_shape(kind)
        # fill by agent color
        fill = self.agent_colors.get(agent, self.cc["default_other_color"])
        style["style"] = style.get("style", "") + ",filled"
        style["fillcolor"] = fill

        # Apply previous_period style for lag variables (ending with *)
        if name.endswith("*"):
            style.update(self.ns.get("previous_period", {}))

        # font etc from default
        node = pydot.Node(name, **style)
        self.nodes[name] = node
        return node

    def create_graph(self, title=None):
        """
        Return a PyDot graph visualization of this block.

        Base style configuration is in model_visualization_config.yaml

        Parameters
        -----------
        title : string, optional
            A title for the generated graph.

        Returns
        -----------

        graph: pydot.core.Dot
            A PyDot graph representation of the model.
        """
        # 1) New graph
        gconf = self.gc
        graph = pydot.Dot(graph_type=gconf["graph_type"])
        # title
        if title is None:
            if self.title is not None:
                title = self.title
            else:
                title = gconf["title"]
        graph.set_label(title)
        graph.set_labelloc("t")
        # layout
        graph.set("rankdir", self.gl["rankdir"])
        graph.set("nodesep", str(self.gl["node_padding"]))

        # 2) pre-create nodes (incl. ALL prev-period nodes for lag edges)
        for var in self.meta:
            self._make_node(var)

        # Create previous period nodes for lag edges ONLY if not already provided by analyzer
        for src, tgt in self.edges.get("lag", []):
            prev = f"{src}*"
            # Only create metadata if analyzer hasn't already provided it
            if prev not in self.meta:
                # Inherit properties from current period node
                src_meta = self.meta.get(src, {})
                self.meta[prev] = {
                    "kind": src_meta.get("kind", "state"),
                    "agent": src_meta.get("agent", "other"),
                    "observed": False,
                    "plate": src_meta.get("plate"),  # Inherit plate from source
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
            # pydot.Cluster automatically adds "cluster_" prefix to graph_name
            sg = pydot.Cluster(
                graph_name=agent,  # Don't add "cluster_" prefix - pydot does it automatically
                label=f"{info['size']} {lbl}",
                labeljust="r",
                style=self.gl["cluster_style"],
                fillcolor=self.gl["cluster_fillcolor"],
                fontsize=str(self.gl["plate_fontsize"]),
            )
            graph.add_subgraph(sg)
            plate_subs[agent] = sg

        # 4) add nodes to plates or root based on their plate metadata
        for name, m in self.meta.items():
            node = self.nodes[name]
            pl = m.get("plate")
            if pl and pl in plate_subs:
                plate_subs[pl].add_node(node)
            else:
                graph.add_node(node)

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

    def display(self):
        display(SVG(self.create_graph().create_svg()))

    def get_image(self):
        graph = self.create_graph()
        svg_content = graph.create_svg()

        # Convert SVG to PNG in memory
        if isinstance(svg_content, str):
            svg_content = svg_content.encode("utf-8")

        try:
            import cairosvg

            png_data = cairosvg.svg2png(bytestring=svg_content)

            # Load PNG into matplotlib
            img = mpimg.imread(BytesIO(png_data), format="png")
        except (ImportError, OSError):
            print(
                "Unable to import cairosvg. This is likely because Cairo is not installed. If using windows, install GTK runtime."
            )
            img = None

        return img, svg_content
