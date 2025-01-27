from dataclasses import dataclass
import networkx as nx

class NodeType:
    # Node types (rooms/doors) and their IDs from HouseGAN++
    # fmt: off
    LIVING_ROOM   = 0
    KITCHEN       = 1
    BEDROOM       = 2
    BATHROOM      = 3
    BALCONY       = 4
    ENTRANCE      = 5
    DINING_ROOM   = 6
    STUDY_ROOM    = 7
    STORAGE       = 9
    FRONT_DOOR    = 14
    UNKNOWN       = 15
    INTERIOR_DOOR = 16
    # fmt: on

    # This is what the model expects
    NUM_NODE_TYPES = 18

    @classmethod
    def is_door(cls, node: int) -> bool:
        """
        Check if the given node type corresponds to a door.

        Args:
            node (int): Node type identifier.

        Returns:
            bool: True if the node represents a door, False otherwise.
        """
        return node in [cls.FRONT_DOOR, cls.INTERIOR_DOOR]

    @classmethod
    def is_room(cls, node: int) -> bool:
        """
        Check if the given node type corresponds to a room.

        Args:
            node (int): Node type identifier.

        Returns:
            bool: True if the node represents a room, False otherwise.
        """
        return not cls.is_door(node)


# fmt: off
NODE_COLOR = {
    NodeType.LIVING_ROOM   : "#EE4D4D",
    NodeType.KITCHEN       : "#C67C7B",
    NodeType.BEDROOM       : "#FFD274",
    NodeType.BATHROOM      : "#BEBEBE",
    NodeType.BALCONY       : "#BFE3E8",
    NodeType.ENTRANCE      : "#7BA779",
    NodeType.DINING_ROOM   : "#E87A90",
    NodeType.STUDY_ROOM    : "#FF8C69",
    NodeType.STORAGE       : "#1F849B",
    NodeType.FRONT_DOOR    : "#727171",
    NodeType.UNKNOWN       : "#785A67",
    NodeType.INTERIOR_DOOR : "#D3A2C7",
}
# fmt: on

# fmt: off
NODE_NAME = {
    NodeType.LIVING_ROOM   : "L",
    NodeType.KITCHEN       : "K",
    NodeType.BEDROOM       : "R",
    NodeType.BATHROOM      : "H",
    NodeType.BALCONY       : "A",
    NodeType.ENTRANCE      : "/",
    NodeType.DINING_ROOM   : "/",
    NodeType.STUDY_ROOM    : "/",
    NodeType.STORAGE       : "/",
    NodeType.FRONT_DOOR    : ":F",
    NodeType.UNKNOWN       : "/",
    NodeType.INTERIOR_DOOR : ":D",
}
# fmt: on

@dataclass
class LayoutGraph:
    nodes: list[int]
    """
    List of node type IDs.
    """

    edges: list[tuple[int, int]]
    """
    List of undirected edges between nodes. Each edge tuple contains index into
    the `nodes` list i.e the tuple (a, b) in this list tells
    that node[a] is a neighbour of node[b]
    """

    def to_networkx(self):
        # nodes_tuples = list(enumerate(self.nodes))
        G = nx.Graph()

        G.add_nodes_from([
            (i, { "node_type": n })
            for i, n in enumerate(self.nodes)
        ])
        G.add_edges_from(self.edges)

        return G

    def draw(self):
        G = self.to_networkx()

        nx.draw(
            G,
            nx.kamada_kawai_layout(G),
            node_size=1000,
            node_color=self._nodelist(lambda n: NODE_COLOR[n]),
            with_labels=True,
            labels=self._nodedict(lambda n: NODE_NAME[n]),
            font_color="black",
            font_weight="bold",
            font_size=14,
            edge_color="#b9c991",
            width=2.0,
        )


    def _nodelist(self, func):
        return [func(node) for node in self.nodes]

    def _nodedict(self, func):
        return { i: func(node) for i, node in enumerate(self.nodes) }
