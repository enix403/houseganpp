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
from PIL import Image, ImageDraw


class FloorplanGraphDataset(Dataset):
    def __init__(self, data_path, transform=None, target_set=8, split="test"):
        super(Dataset, self).__init__()

        self.split = split
        self.subgraphs = []
        self.target_set = target_set

        with open(data_path, "r") as f1:
            lines = f1.readlines()
            lines = list(map(lambda x: x.strip(), lines))

        for line in lines:
            rms_type, fp_eds, rms_bbs, eds_to_rms, eds_to_rms_tmp = reader(
                line
            )

            self.subgraphs.append([
                rms_type,
                rms_bbs,
                fp_eds,
                eds_to_rms,
                eds_to_rms_tmp,
            ])

        self.transform = transform

    def __len__(self):
        return len(self.subgraphs)

    def __getitem__(self, index):

        graph = self.subgraphs[index]

        rms_type = graph[0]
        rms_bbs = graph[1]
        fp_eds = graph[2]
        eds_to_rms = graph[3]
        eds_to_rms_tmp = graph[4]

        # rms_bbs = np.array(rms_bbs)
        # fp_eds = np.array(fp_eds)

        # extract boundary box and centralize
        # ------
        tl = np.min(rms_bbs[:, :2], 0)
        br = np.max(rms_bbs[:, 2:], 0)
        shift = (tl + br) / 2.0 - 0.5
        rms_bbs[:, :2] -= shift
        rms_bbs[:, 2:] -= shift
        fp_eds[:, :2] -= shift
        fp_eds[:, 2:] -= shift
        # tl -= shift
        # br -= shift
        # eds_to_rms_tmp = []
        # for l in range(len(eds_to_rms)):
        #     eds_to_rms_tmp.append([eds_to_rms[l][0]])
        # ------

        # build input graph
        graph_nodes, graph_edges, rooms_mks = self.build_graph(
            rms_type, fp_eds, eds_to_rms
        )

        # convert to tensor
        graph_nodes = one_hot_embedding(graph_nodes)[:, 1:]
        graph_nodes = torch.FloatTensor(graph_nodes)
        graph_edges = torch.LongTensor(graph_edges)
        rooms_mks = torch.FloatTensor(rooms_mks)
        rooms_mks = self.transform(rooms_mks)

        return rooms_mks, graph_nodes, graph_edges

    def build_graph(self, rms_type, fp_eds, eds_to_rms):

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
                fp_mk[mask_on_at] = k + 1

        # trick to remove overlap
        for k in range(len(nodes)):
            if rms_type[k] != 15 and rms_type[k] != 17:
                rm_arr = np.zeros((out_size, out_size))
                inds = np.where(fp_mk == k + 1)
                rm_arr[inds] = 1.0
                rms_masks[k] = rm_arr

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

        nodes = np.array(nodes)
        return nodes, triples, rms_masks


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


def floorplan_collate_fn(batch):

    all_rooms_mks, all_nodes, all_edges = [], [], []
    all_node_to_sample, all_edge_to_sample = [], []
    node_offset = 0
    eds_sets = []
    for i, (rooms_mks, nodes, edges) in enumerate(batch):
        O, T = nodes.size(0), edges.size(0)
        all_rooms_mks.append(rooms_mks)
        all_nodes.append(nodes)
        # eds_sets.append(eds_set)
        edges = edges.clone()
        if edges.shape[0] > 0:
            edges[:, 0] += node_offset
            edges[:, 2] += node_offset
            all_edges.append(edges)
        all_node_to_sample.append(torch.LongTensor(O).fill_(i))
        all_edge_to_sample.append(torch.LongTensor(T).fill_(i))
        node_offset += O
    all_rooms_mks = torch.cat(all_rooms_mks, 0)
    all_nodes = torch.cat(all_nodes)
    if len(all_edges) > 0:
        all_edges = torch.cat(all_edges)
    else:
        all_edges = torch.tensor([])
    all_node_to_sample = torch.cat(all_node_to_sample)
    all_edge_to_sample = torch.cat(all_edge_to_sample)

    return all_rooms_mks, all_nodes, all_edges, all_node_to_sample, all_edge_to_sample


# Returns (as tuple)
#   rms_type: "room_type" from file
#   rms_bbs: (R, 4): "boxes" from file, but centralized
#   fp_eds: (E, 4): "edges" from file, but centralized
#   eds_to_rms: list[list[int]] "ed_rm" from file
#   eds_to_rms_tmp: eds_to_rms but the inner list truncated to length 1
#                   i.e eds_to_rms_tmp[i][0] == eds_to_rms[i][0] for every i
def reader(filename):
    with open(filename) as f:
        # open file
        info = json.load(f)

        # read from file
        # rms_bbs = np.asarray(info["boxes"])
        # fp_eds = info["edges"]
        # rms_type = info["room_type"]
        # eds_to_rms = info["ed_rm"]

        rms_type = info["room_type"]
        rms_bbs = np.array(info["boxes"])
        fp_eds = np.array(info["edges"])
        eds_to_rms = info["ed_rm"]

        # Count rooms that are not equal to 17 (interior_door)
        # s_r = 0
        # for rmk in range(len(rms_type)):
        #     if rms_type[rmk] != 17:
        #         s_r = s_r + 1

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

        # tl -= shift
        # br -= shift

        # eds_to_rms_tmp = []
        # for l in range(len(eds_to_rms)):
            # eds_to_rms_tmp.append([eds_to_rms[l][0]])

        # eds_to_rms but the inner list truncated to length 1
        eds_to_rms_tmp = [ar[:1] for ar in eds_to_rms]

        return rms_type, fp_eds, rms_bbs, eds_to_rms, eds_to_rms_tmp



