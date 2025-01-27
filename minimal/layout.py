from dataclasses import dataclass


class NodeType:
    # Node types (rooms/doors) and their IDs from HouseGAN++
    LIVING_ROOM = 0
    KITCHEN = 1
    BEDROOM = 2
    BATHROOM = 3
    BALCONY = 4
    ENTRANCE = 5
    DINING_ROOM = 6
    STUDY_ROOM = 7
    STORAGE = 9
    FRONT_DOOR = 14
    UNKNOWN = 15
    INTERIOR_DOOR = 16

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
