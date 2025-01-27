from dataclasses import dataclass

import torch
import torch.nn.functional as F

from minimal.arch import Generator

# -----------------------------

PRETRAINED_PATH = "./checkpoints/pretrained.pth"

model = Generator()
model.load_state_dict(
    torch.load(PRETRAINED_PATH, map_location=torch.device("cpu")), strict=True
)
model = model.eval()

# -----------------------------

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

# -----------------------------

def _prepare_fixed_masks(masks, idx_fixed):
    num_nodes = masks.shape[0]

    # (R, 64, 64)
    label_bg = torch.zeros_like(masks)
    masks = masks.clone()

    idx_not_fixed = [k for k in range(num_nodes) if k not in idx_fixed],

    # Label the fixed nodes
    label_bg[idx_fixed] = 1.0

    # Label the unfixed nodes, as well as clear
    # out their mask
    label_bg[idx_not_fixed] = 0.0
    masks[idx_not_fixed] = -1.0

    return torch.stack([masks, label_bg], dim=1)

@torch.no_grad()
def _predict_masks(nodes_enc, edges_enc, prev_masks=None, idx_fixed=[]):

    # Input is: 
    #       z = (R, 128)
    # given_m = (R, 2, 64, 64)
    # given_y = (R, 18)
    # given_w = (E(R), 3)

    num_nodes = nodes_enc.shape[0]

    z = torch.randn(num_nodes, 128)

    if prev_masks is None:
        prev_masks = torch.zeros((num_nodes, 64, 64)) - 1.0

    # (R, 2, 64, 64)
    fixed_masks = _prepare_fixed_masks(prev_masks, idx_fixed)

    next_masks = model(z, fixed_masks, nodes_enc, edges_enc)
    return next_masks.detach()

# -----------------------------

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

# -----------------------------

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
    masks = _predict_masks(
        nodes_enc, edges_enc,
        prev_masks=None,
        idx_fixed=[]
    )

    for i in range(num_iters):
        fixed_nodes = unique_nodes[:i]

        idx_fixed = [k for k in range(len(nodes)) if nodes[k] in fixed_nodes]

        # Iteratively improve masks
        masks = _predict_masks(
            nodes_enc, edges_enc,
            prev_masks=masks,
            idx_fixed=idx_fixed
        )

    return masks

# --------------------
