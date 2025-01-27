
from dataclasses import dataclass

# import torch
# import torch.nn.functional as F

# nds = F.one_hot(torch.tensor([4,1]), num_classes=NodeType.NUM_NODE_TYPES).float()

# print(F.one_hot(
#     torch.tensor([NodeType.LIVING_ROOM, NodeType.STUDY_ROOM]),
#     num_classes=NodeType.NUM_NODE_TYPES
# ).float())

class NodeType:
    # Node types (rooms/doors) and their IDs from HouseGAN++
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

    # This is what the model expects
    NUM_NODE_TYPES = 18

    @classmethod
    def is_door(cls, node: int) -> bool:
        return node in [cls.FRONT_DOOR, cls.INTERIOR_DOOR]

    @classmethod
    def is_room(cls, node: int) -> bool:
        return not cls.is_door(node)

@dataclass
class LayoutGraph:
    nodes: list[int]
    edges: list[(int, int)]

def generate_plan_v2(layout_graph: LayoutGraph, num_iters: int=10):
    nodes = layout_graph.nodes
    edges = layout_graph.edges

    nodes_enc = F.one_hot(
        torch.tensor(nodes),
        num_classes=NodeType.NUM_NODE_TYPES
    ).float()
    edges_enc = _make_edge_triplets(layout_graph)

    unique_nodes = sorted(list(set(nodes)))

    # Generate initial mask
    masks = _infer_step(
        nodes_enc, edges_enc,
        masks=None,
        fixed_nodes=[]
    )

    for i in range(num_iters):
        fixed_nodes_ids = unique_nodes[:i]

        fixed_nodes = [k for k in range(len(nodes)) if nodes[k] in fixed_nodes_ids]

        # Iteratively improve masks
        masks = _infer_step(
            nodes_enc, edges_enc,
            masks=masks,
            fixed_nodes=fixed_nodes
        )

    return masks

# --------------------

def _make_edge_triplets(layout_graph: LayoutGraph):
    n = len(layout_graph.nodes)
    edges = set(layout_graph.edges)

    triplets: list[(int, int, int)] = []

    for a in range(n):
        for b in range(a + 1, n):
            is_joined = ((a, b) in edges) or ((b, a) in edges)
            triplets.append((
                a,
                1 if is_joined else -1
                b,
            ))

    return torch.tensor(triplets, dtype=torch.long)