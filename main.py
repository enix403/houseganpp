import os
import numpy as np

import torch
from torchvision.utils import save_image

from minimal.imaging import draw_plan, draw_graph
from minimal.layout import LayoutGraph
from minimal.gen import generate_plan, _make_edge_triplets

print("Starting generation")

g = LayoutGraph([4, 2, 2, 3, 1, 0, 16, 16, 16, 16, 16, 14], [(0, 5), (0, 7), (1, 5), (1, 6), (2, 5), (2, 8), (3, 5), (3, 9), (4, 5), (4, 10), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11)])

# masks = generate_plan(g)

# # save final floorplans
# img = draw_plan(masks, g.nodes)
# img = torch.tensor(np.array(img).transpose((2, 0, 1))) / 255.0
# save_image(img, "dump/fp_final_0.png", nrow=1, normalize=False)

# print("Done")

# -----

# G, graph_im = draw_graph([
#     np.array(g.nodes),
#     _make_edge_triplets(g).numpy()
# ])
# graph_im.save("dump/graph_0.png")

