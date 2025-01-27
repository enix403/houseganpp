from typing import Union, Iterable
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

class LayoutGraph:
    nodes: list[int]
    """
    List of node type IDs.
    """

    edges: set[tuple[int, int]]
    """
    List of undirected edges between nodes. Each edge tuple contains index into
    the `nodes` list i.e the tuple (a, b) in this list tells
    that node[a] is a neighbour of node[b]
    """

    def __init__(
        self,
        nodes: Iterable[int],
        edges: Iterable[tuple[int, int]]
    ):
        self.nodes = list(nodes)
        self.edges = set(edges)

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

    def has_edge(self, a: int, b: int):
        """Check if rooms a and b are connected"""
        return (a, b) in self.edges or (b, a) in self.edges

    def _find_door_between(self, a: int, b: int):
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


class LayoutGraphBuilderNode:
    type: int

    def __init__(self, type: int):
        self.type = type
        self.index = -1


class LayoutGraphBuilder:
    nodes: list[LayoutGraphBuilderNode]
    edges: list[tuple[LayoutGraphBuilderNode, LayoutGraphBuilderNode]]

    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, type):
        node = LayoutGraphBuilderNode(type)
        self.nodes.append(node)
        return node

    def add_edge(
        self,
        a: Union[LayoutGraphBuilderNode, int],
        b: Union[LayoutGraphBuilderNode, int],
    ):
        if isinstance(a, int):
            a = self.add_node(a)

        if isinstance(b, int):
            b = self.add_node(b)

        self.edges.append((a, b))

        return (a, b)


    def build(self) -> LayoutGraph:
        self.nodes.sort(key=lambda n: n.type)

        for i, node in enumerate(self.nodes):
            node.index = i

        return LayoutGraph(
            map(lambda node: node.type, self.nodes),
            map(lambda e: (e[0].index, e[1].index), self.edges)
        )