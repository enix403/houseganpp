#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

_transform_box = transforms.Normalize(mean=[0.5], std=[0.5])

DATA_PATH = "./data/sample_list.txt"


class FloorplanGraphDataset(Dataset):
    def __init__(self):
        super().__init__()

        with open(DATA_PATH, "r") as f1:
            lines = f1.readlines()

        # [(rms_type, fp_eds, eds_to_rms)]
        self.subgraphs = [reader(line.strip()) for line in lines]

    def __len__(self):
        return len(self.subgraphs)

    def __getitem__(self, index):

        rms_type, _, fp_eds, eds_to_rms = self.subgraphs[index]

        triples, rms_masks = build_graph(rms_type, fp_eds, eds_to_rms)

        graph_nodes = torch.FloatTensor(one_hot_embedding(np.array(rms_type))[:, 1:])
        graph_edges = torch.LongTensor(triples)

        # transform converts range [0, 1] to [-1, 1]
        rooms_mks = _transform_box(torch.FloatTensor(rms_masks))

        return rooms_mks, graph_nodes, graph_edges


def build_graph(rms_type, fp_eds, eds_to_rms):

    nodes = rms_type

    # ------------

    rms_masks = []
    im_size = 256
    out_size = 64
    fp_mk = np.zeros((out_size, out_size))

    for k in range(len(nodes)):

        # Index of edges from eds_to_rms which have node k as their first node
        eds = [i for i, inner in enumerate(eds_to_rms) if inner[0] == k]

        edges = fp_eds[eds]
        poly = make_sequence(edges)[0]
        poly = [(im_size * x, im_size * y) for x, y in poly]

        if len(poly) < 2:
            print("Empty room")
            exit(0)

        mask_canvas = Image.new("L", (im_size, im_size))
        ImageDraw.Draw(mask_canvas).polygon(poly, fill="white")

        mask = np.array(mask_canvas.resize((out_size, out_size)))

        # The resizing operation above may blur out some pixels
        # Ensure that the mask only contains 0 and 1, and nothing else
        mask_on_at = np.where(mask > 0)
        mask[mask_on_at] = 1.0

        rms_masks.append(mask)

        if rms_type[k] != 15 and rms_type[k] != 17:
            # It may overwrite some previous mask value
            # This will be corrected later
            fp_mk[mask_on_at] = k + 1

    # trick to remove overlap
    for k in range(len(nodes)):
        if rms_type[k] != 15 and rms_type[k] != 17:
            inds = np.where(fp_mk == k + 1)

            temp_canvas = np.zeros((out_size, out_size))
            temp_canvas[inds] = 1.0

            rms_masks[k] = temp_canvas

    rms_masks = np.array(rms_masks)

    # ------------

    # a 3-length list for each pair of nodes (a, b) such
    # that each item is
    #   [a,  1, b] if a is connected to b
    #   [a, -1, b] if a is not connected to b
    triples = []

    for k in range(len(nodes)):
        for l in range(k + 1, len(nodes)):
            is_adjacent = False
            for edge in eds_to_rms:
                if (l in edge) and (k in edge):
                    is_adjacent = True
                    break

            triples.append([k, 1 if is_adjacent else -1, l])

    triples = np.array(triples)

    # ------------

    return triples, rms_masks


def make_sequence(edges):
    polys = []

    v_curr = tuple(edges[0][:2])
    e_ind_curr = 0
    e_visited = [0]
    seq_tracker = [v_curr]
    find_next = False

    while len(e_visited) < len(edges):
        if find_next == False:
            if v_curr == tuple(edges[e_ind_curr][2:]):
                v_curr = tuple(edges[e_ind_curr][:2])
            else:
                v_curr = tuple(edges[e_ind_curr][2:])
            find_next = True
        else:
            # look for next edge
            for k, e in enumerate(edges):
                if k not in e_visited:
                    if v_curr == tuple(e[:2]):
                        v_curr = tuple(e[2:])
                        e_ind_curr = k
                        e_visited.append(k)
                        break
                    elif v_curr == tuple(e[2:]):
                        v_curr = tuple(e[:2])
                        e_ind_curr = k
                        e_visited.append(k)
                        break

        # extract next sequence
        if v_curr == seq_tracker[-1]:
            polys.append(seq_tracker)
            for k, e in enumerate(edges):
                if k not in e_visited:
                    v_curr = tuple(edges[0][:2])
                    seq_tracker = [v_curr]
                    find_next = False
                    e_ind_curr = k
                    e_visited.append(k)
                    break
        else:
            seq_tracker.append(v_curr)
    polys.append(seq_tracker)

    return polys


def one_hot_embedding(labels, num_classes=19):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    # print(" label is",labels)
    return y[labels]


def reader(filename):
    with open(filename) as f:
        # open file
        info = json.load(f)

        # read from file
        rms_type = info["room_type"]
        rms_bbs = np.array(info["boxes"])
        fp_eds = np.array(info["edges"])
        eds_to_rms = info["ed_rm"]

        # Convert bounding boxes from range [0,256] to range [0,1]
        rms_bbs = np.array(rms_bbs) / 256.0
        fp_eds = np.array(fp_eds) / 256.0

        # discard last 2 number of fp_eds
        # new shape: (*, 4)
        fp_eds = fp_eds[:, :4]

        # TL and BR of the global bounding box of the masks
        tl = np.min(rms_bbs[:, :2], 0)
        br = np.max(rms_bbs[:, 2:], 0)

        # Shift the center of the overall bounding box to be at (0.5, 0.5)
        # i.e centralize the floor plan
        shift = (tl + br) / 2.0 - 0.5
        rms_bbs[:, :2] -= shift
        rms_bbs[:, 2:] -= shift
        fp_eds[:, :2] -= shift
        fp_eds[:, 2:] -= shift

        return rms_type, rms_bbs, fp_eds, eds_to_rms
