import os
import numpy as np

import torch
from torchvision.utils import save_image

from minimal.arch import Generator
from minimal.dataset import FloorplanGraphDataset
from minimal.utils import init_input, draw_masks, draw_graph

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

def _infer(graph, model, prev_state=None):
    z, given_masks_in, given_nds, given_eds = init_input(graph, prev_state)

    with torch.no_grad():
        masks = model(z, given_masks_in, given_nds, given_eds)
        masks = masks.detach().cpu().numpy()

    return masks

NUM_ITERS = 10

fp_dataset = FloorplanGraphDataset(DATA_PATH)

i = 0
sample = next(iter(fp_dataset))

# mks (rooms_mks)   (R, 64, 64) = GT segmentation mask per room
# nds (graph_nodes) (R, 18) = one hot encoding per room
# eds (graph_edges) (E, 3) = per edge [node_1, -1 / 1, node_2]
_, nds, eds = sample
graph = [nds, eds]

rms_type_z = np.where(nds.detach().cpu() == 1)[1]
_types = sorted(list(set(rms_type_z)))
selected_types = [_types[:k+1] for k in range(NUM_ITERS)]

# -------

state = { "masks": None, "fixed_nodes": [] }

# (R, 64, 64): mask per room
masks = _infer(graph, model, state)

# generate per room type
for _types in selected_types:

    # list[int], indexes of rooms to fix
    fixed_nodes = np.concatenate([
        np.where(rms_type_z == _t)[0]
        for _t in _types
    ])

    state = { "masks": masks, "fixed_nodes": fixed_nodes }
    masks = _infer(graph, model, state)

# -----

# save final floorplans
# imk = draw_masks(masks.copy(), rms_type_z)
# imk = torch.tensor(np.array(imk).transpose((2, 0, 1))) / 255.0
# save_image(imk, "./{}/fp_final_{}.png".format(OUT_PATH, i), nrow=1, normalize=False)

# -----

# rms_type_z = np.where(nds.detach().cpu() == 1)[-1]
# true_graph_obj: networkx graph
# true_graph_obj, graph_im = draw_graph([rms_type_z, eds.detach().cpu().numpy()])
# graph_im.save("./{}/graph_{}.png".format(OUT_PATH, i))  # save graph

