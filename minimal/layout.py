import copy
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
    NodeType.ENTRANCE      : "E",
    NodeType.DINING_ROOM   : "D",
    NodeType.STUDY_ROOM    : "S",
    NodeType.STORAGE       : "T",
    NodeType.FRONT_DOOR    : ":f",
    NodeType.UNKNOWN       : "/",
    NodeType.INTERIOR_DOOR : ":d",
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

    def clone(self):
        return LayoutGraph(
            copy.deepcopy(self.nodes),
            copy.deepcopy(self.edges),
        )

    def to_networkx(self):
        G = nx.Graph()

        G.add_nodes_from([
            (i, { "node_type": n })
            for i, n in enumerate(self.nodes)
        ])
        G.add_edges_from(self.edges)

        return G

    def has_edge(self, a, b):
        """Check if rooms a and b are connected"""
        return (a, b) in self.edges or (b, a) in self.edges

    def _find_door_between(self, a, b):
        """Find a door connecting room a and b"""

        for i, node in enumerate(self.nodes):
            if node == NodeType.INTERIOR_DOOR:
                if self.has_edge(a, i) and self.has_edge(b, i):
                    return i

        return -1

    def ensure_door_connections(self):
        nodes = copy.copy(self.nodes)
        edges = copy.deepcopy(self.edges)

        adjacent_rooms = []

        for (a, b) in edges:
            na = nodes[a]
            nb = nodes[b]

            if NodeType.is_room(na) and NodeType.is_room(nb):
                # Add doors only between rooms that are not
                # already connected
                if self._find_door_between(a, b) == -1:
                    adjacent_rooms.append((a, b))

        for a, b in adjacent_rooms:
            door_idx = len(nodes) 
            nodes.append(NodeType.INTERIOR_DOOR)

            edges.append((a, door_idx))
            edges.append((b, door_idx))

        self.nodes = nodes
        self.edges = edges

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
1