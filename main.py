import os
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

from minimal.arch import Generator

from minimal.dataset import (
    FloorplanGraphDataset,
    floorplan_collate_fn
)

from minimal.utils import (
    init_input,
    draw_masks,
    draw_graph
)

PRETRAINED_PATH = "./checkpoints/pretrained.pth"
DATA_PATH = "./data/sample_list.txt"
OUT_PATH = "./dump"

os.makedirs(OUT_PATH, exist_ok=True)

# Initialize generator and discriminator
model = Generator()
model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=torch.device('cpu')), strict=True)
model = model.eval()

# initialize dataset iterator
fp_dataset_test = FloorplanGraphDataset(
    DATA_PATH,
    transforms.Normalize(mean=[0.5], std=[0.5]),
    split='test'
)

fp_loader = torch.utils.data.DataLoader(
    fp_dataset_test, 
    batch_size=1, 
    shuffle=False,
    collate_fn=floorplan_collate_fn
)

# run inference
def _infer(graph, model, prev_state=None):
    
    # configure input to the network
    z, given_masks_in, given_nds, given_eds = init_input(graph, prev_state)
    # run inference model
    with torch.no_grad():
        masks = model(z, given_masks_in, given_nds, given_eds)
        masks = masks.detach().cpu().numpy()
    return masks

# for i, sample in enumerate(fp_loader):
if True:
    i, sample = next(enumerate(fp_loader))

    # mks (R, 64, 64) = GT segmentation mask per room
    # nds (R, 18) = one hot encoding per room
    # eds (E, 3) = per edge [node_1, -1 / 1 ???, node_2]
    mks, nds, eds, _, _ = sample

    # (R,) undo one hot encoding (0-index based)
    real_nodes = np.where(nds.detach().cpu()==1)[-1]
    graph = [nds, eds]
    # true_graph_obj: networkx graph
    # graph_im: PIL graph image
    true_graph_obj, graph_im = draw_graph([real_nodes, eds.detach().cpu().numpy()])
    graph_im.save('./{}/graph_{}.png'.format(OUT_PATH, i)) # save graph

    # add room types incrementally
    _types = sorted(list(set(real_nodes)))
    selected_types = [_types[:k+1] for k in range(10)]
    _round = 0
    
    # initialize layout
    state = {
        'masks': None,
        'fixed_nodes': []
    }

    masks = _infer(graph, model, state)
    # im0 = draw_masks(masks.copy(), real_nodes)
    # im0 = torch.tensor(np.array(im0).transpose((2, 0, 1)))/255.0 
    # # visualize init image
    # save_image(im0, './{}/fp_init_{}.png'.format(OUT_PATH, i), nrow=1, normalize=False)

    # generate per room type
    for _iter, _types in enumerate(selected_types):
        _fixed_nds = np.concatenate([np.where(real_nodes == _t)[0] for _t in _types]) \
            if len(_types) > 0 else np.array([]) 
        state = {'masks': masks, 'fixed_nodes': _fixed_nds}
        masks = _infer(graph, model, state)
        
    # save final floorplans
    imk = draw_masks(masks.copy(), real_nodes)
    imk = torch.tensor(np.array(imk).transpose((2, 0, 1)))/255.0 
    save_image(imk, './{}/fp_final_{}.png'.format(OUT_PATH, i), nrow=1, normalize=False)
    
