import os
import numpy as np

import torch
from torchvision.utils import save_image

from minimal.dataset import FloorplanGraphDataset
from minimal.utils import draw_plan, draw_graph
from minimal.gen import generate_plan

OUT_PATH = "./output"

print("Starting generation")

fp_dataset = FloorplanGraphDataset()
sample = next(iter(fp_dataset))
_, nds, eds = sample

masks = generate_plan(nds, eds)

# save final floorplans
imk = draw_plan(masks, np.where(nds == 1)[1])
imk = torch.tensor(np.array(imk).transpose((2, 0, 1))) / 255.0
save_image(imk, "{}/fp_final_0.png".format(OUT_PATH), nrow=1, normalize=False)

print("Done")

# -----

# rms_type_z = np.where(nds.detach().cpu() == 1)[-1]
# true_graph_obj: networkx graph
# true_graph_obj, graph_im = draw_graph([rms_type_z, eds.detach().cpu().numpy()])
# graph_im.save("./{}/graph_{}.png".format(OUT_PATH, i))  # save graph

