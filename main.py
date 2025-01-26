import os
import numpy as np

import torch
from torchvision.utils import save_image

from minimal.arch import Generator
from minimal.dataset import FloorplanGraphDataset
from minimal.utils import draw_masks, draw_graph

PRETRAINED_PATH = "./checkpoints/pretrained.pth"
DATA_PATH = "./data/sample_list.txt"
OUT_PATH = "./dump"

# os.makedirs(OUT_PATH, exist_ok=True)

# Initialize generator and discriminator
model = Generator()
model.load_state_dict(
    torch.load(PRETRAINED_PATH, map_location=torch.device("cpu")), strict=True
)
model = model.eval()

def prepare_fixed_masks(masks, fixed_nodes):
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
    fixed_masks = prepare_fixed_masks(masks, fixed_nodes)

    next_masks = model(z, fixed_masks, nds, eds)
    return next_masks.detach()

NUM_ITERS = 10

fp_dataset = FloorplanGraphDataset(DATA_PATH)

i = 0
sample = next(iter(fp_dataset))

# mks (rooms_mks)   (R, 64, 64) = GT segmentation mask per room
# nds (graph_nodes) (R, 18) = one hot encoding per room
# eds (graph_edges) (E, 3) = per edge [node_1, -1 / 1, node_2]
_, nds, eds = sample

rms_type_z = np.where(nds == 1)[1]
_types = sorted(list(set(rms_type_z)))
selected_types = [_types[:k+1] for k in range(NUM_ITERS)]

# -------

print("Starting generation")

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
# -----

# save final floorplans
imk = draw_masks(masks.copy(), rms_type_z)
imk = torch.tensor(np.array(imk).transpose((2, 0, 1))) / 255.0
save_image(imk, "./{}/fp_final_{}.png".format(OUT_PATH, i), nrow=1, normalize=False)

print("Done")

# -----

# rms_type_z = np.where(nds.detach().cpu() == 1)[-1]
# true_graph_obj: networkx graph
# true_graph_obj, graph_im = draw_graph([rms_type_z, eds.detach().cpu().numpy()])
# graph_im.save("./{}/graph_{}.png".format(OUT_PATH, i))  # save graph

