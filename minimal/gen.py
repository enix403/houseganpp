import torch
import numpy as np

from minimal.arch import Generator

PRETRAINED_PATH = "./checkpoints/pretrained.pth"

model = Generator()
model.load_state_dict(
    torch.load(PRETRAINED_PATH, map_location=torch.device("cpu")), strict=True
)
model = model.eval()

def _prepare_fixed_masks(masks, fixed_nodes):
    # (R, 64, 64)
    masks = masks.clone()

    ind_fixed = torch.tensor(fixed_nodes, dtype=torch.long)
    ind_not_fixed = torch.tensor(
        [k for k in range(masks.shape[0]) if k not in ind_fixed],
        dtype=torch.long
    )

    masks[ind_not_fixed] = -1.0
    
    label_bg = torch.zeros_like(masks)
    label_bg[ind_fixed] = 1.0
    label_bg[ind_not_fixed] = 0.0

    return torch.stack([masks, label_bg], dim=1)

@torch.no_grad()
def _infer(nds, eds, masks=None, fixed_nodes=[]):

    # Input is: 
    #       z = (R, 128)
    # given_m = (R, 2, 64, 64)
    # given_y = (R, 18)
    # given_w = (E(R), 3)

    z = torch.randn(len(nds), 128)

    if masks is None:
        masks = torch.zeros((nds.shape[0], 64, 64)) - 1.0

    # (R, 2, 64, 64)
    fixed_masks = _prepare_fixed_masks(masks, fixed_nodes)

    next_masks = model(z, fixed_masks, nds, eds)
    return next_masks.detach()


def generate_plan(nds, eds, num_iters=10):
    # nds (graph_nodes) (R, 18) = one hot encoding per room
    # eds (graph_edges) (E, 3) = per edge [node_1, -1 / 1, node_2]

    rms_type_z = np.where(nds == 1)[1]
    _types = sorted(list(set(rms_type_z)))
    selected_types = [_types[:k+1] for k in range(num_iters)]

    # (R, 64, 64): mask per room
    masks = _infer(nds, eds, masks=None, fixed_nodes=[])

    # generate per room type
    for _types in selected_types:

        # list[int], indexes of rooms to fix
        fixed_nodes = np.concatenate([
            np.where(rms_type_z == _t)[0]
            for _t in _types
        ])

        masks = _infer(nds, eds, masks=masks, fixed_nodes=fixed_nodes)

    masks = masks.numpy()

    return masks