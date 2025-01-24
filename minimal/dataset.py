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


def filter_graphs(graphs, min_h=0.03, min_w=0.03):
    new_graphs = []
    for g in graphs:

        # retrieve data
        rooms_type = g[0]
        rooms_bbs = g[1]

        # discard broken samples
        check_none = np.sum([bb is None for bb in rooms_bbs])
        check_node = np.sum([nd == 0 for nd in rooms_type])
        if (len(rooms_type) == 0) or (check_none > 0) or (check_node > 0):
            continue

        # update graph
        new_graphs.append(g)
    return new_graphs


class FloorplanGraphDataset(Dataset):
    def __init__(self, data_path, transform=None, target_set=8, split="train"):
        super(Dataset, self).__init__()
        self.split = split
        self.subgraphs = []
        self.target_set = target_set
        f1 = open(data_path, "r")
        lines = f1.readlines()
        h = 0
        for line in lines:
            a = []
            h = h + 1
            if split == "train":
                if h % 1 == 0:
                    with open(line[:-1]) as f2:
                        rms_type, fp_eds, rms_bbs, eds_to_rms, eds_to_rms_tmp = reader(
                            line[:-1]
                        )
                        fp_size = len([x for x in rms_type if x != 15 and x != 17])
                        if fp_size != target_set:
                            a.append(rms_type)
                            a.append(rms_bbs)
                            a.append(fp_eds)
                            a.append(eds_to_rms)
                            a.append(eds_to_rms_tmp)
                            self.subgraphs.append(a)
                self.augment = True
            elif split == "eval":
                if h % 1 == 0:
                    with open(line[:-1]) as f2:
                        rms_type, fp_eds, rms_bbs, eds_to_rms, eds_to_rms_tmp = reader(
                            line[:-1]
                        )
                        fp_size = len([x for x in rms_type if x != 15 and x != 17])
                        if fp_size == target_set:
                            a.append(rms_type)
                            a.append(rms_bbs)
                            a.append(fp_eds)
                            a.append(eds_to_rms)
                            a.append(eds_to_rms_tmp)
                            self.subgraphs.append(a)
                self.augment = False
            elif split == "test":
                if h % 1 == 0:
                    with open(line[:-1]) as f2:
                        rms_type, fp_eds, rms_bbs, eds_to_rms, eds_to_rms_tmp = reader(
                            line[:-1]
                        )
                        a.append(rms_type)
                        a.append(rms_bbs)
                        a.append(fp_eds)
                        a.append(eds_to_rms)
                        a.append(eds_to_rms_tmp)
                        self.subgraphs.append(a)
            else:
                print("ERR")
                exit(0)
        self.transform = transform
        print(len(self.subgraphs))

    def __len__(self):
        return len(self.subgraphs)

    def __getitem__(self, index):

        graph = self.subgraphs[index]
        rms_type = graph[0]
        rms_bbs = graph[1]
        fp_eds = graph[2]
        eds_to_rms = graph[3]
        eds_to_rms_tmp = graph[4]
        rms_bbs = np.array(rms_bbs)
        fp_eds = np.array(fp_eds)

        # extract boundary box and centralize
        tl = np.min(rms_bbs[:, :2], 0)
        br = np.max(rms_bbs[:, 2:], 0)
        shift = (tl + br) / 2.0 - 0.5
        rms_bbs[:, :2] -= shift
        rms_bbs[:, 2:] -= shift
        fp_eds[:, :2] -= shift
        fp_eds[:, 2:] -= shift
        tl -= shift
        br -= shift
        eds_to_rms_tmp = []
        for l in range(len(eds_to_rms)):
            eds_to_rms_tmp.append([eds_to_rms[l][0]])

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

    def make_sequence(self, edges):
        polys = []
        # print(edges)
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
                find_next = not find_next
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

    def build_graph(self, rms_type, fp_eds, eds_to_rms, out_size=64):

        # create edges
        triples = []
        nodes = rms_type

        # encode connections
        for k in range(len(nodes)):
            for l in range(len(nodes)):
                if l > k:
                    is_adjacent = any(
                        [True for e_map in eds_to_rms if (l in e_map) and (k in e_map)]
                    )
                    if is_adjacent:
                        if "train" in self.split:
                            triples.append([k, 1, l])
                        else:
                            triples.append([k, 1, l])
                    else:
                        if "train" in self.split:
                            triples.append([k, -1, l])
                        else:
                            triples.append([k, -1, l])

        # get rooms masks
        eds_to_rms_tmp = []
        for l in range(len(eds_to_rms)):
            eds_to_rms_tmp.append([eds_to_rms[l][0]])

        rms_masks = []
        im_size = 256
        fp_mk = np.zeros((out_size, out_size))

        for k in range(len(nodes)):

            # add rooms and doors
            eds = []
            for l, e_map in enumerate(eds_to_rms_tmp):
                if k in e_map:
                    eds.append(l)

            # draw rooms
            rm_im = Image.new("L", (im_size, im_size))
            dr = ImageDraw.Draw(rm_im)
            for eds_poly in [eds]:
                poly = self.make_sequence(np.array([fp_eds[l][:4] for l in eds_poly]))[
                    0
                ]
                poly = [(im_size * x, im_size * y) for x, y in poly]
                if len(poly) >= 2:
                    dr.polygon(poly, fill="white")
                else:
                    print("Empty room")
                    exit(0)

            # if rms_type[k] == 15 and rms_type[k] == 17:
            #   rm_im = rm_im.filter(ImageFilter.MinFilter(5))

            rm_im = rm_im.resize((out_size, out_size))
            rm_arr = np.array(rm_im)
            inds = np.where(rm_arr > 0)
            rm_arr[inds] = 1.0
            rms_masks.append(rm_arr)

            # deb = Image.fromarray(rm_arr)
            # plt.imwhow(deb)
            # plt.show()
            if rms_type[k] != 15 and rms_type[k] != 17:
                fp_mk[inds] = k + 1

        # trick to remove overlap
        for k in range(len(nodes)):
            if rms_type[k] != 15 and rms_type[k] != 17:
                rm_arr = np.zeros((out_size, out_size))
                inds = np.where(fp_mk == k + 1)
                rm_arr[inds] = 1.0
                rms_masks[k] = rm_arr

        # convert to array
        nodes = np.array(nodes)
        triples = np.array(triples)
        rms_masks = np.array(rms_masks)

        return nodes, triples, rms_masks


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


def reader(filename):
    with open(filename) as f:
        info = json.load(f)
        rms_bbs = np.asarray(info["boxes"])
        fp_eds = info["edges"]
        rms_type = info["room_type"]
        eds_to_rms = info["ed_rm"]
        s_r = 0
        for rmk in range(len(rms_type)):
            if rms_type[rmk] != 17:
                s_r = s_r + 1
        # print("eds_ro",eds_to_rms)
        rms_bbs = np.array(rms_bbs) / 256.0
        fp_eds = np.array(fp_eds) / 256.0
        fp_eds = fp_eds[:, :4]
        tl = np.min(rms_bbs[:, :2], 0)
        br = np.max(rms_bbs[:, 2:], 0)
        shift = (tl + br) / 2.0 - 0.5
        rms_bbs[:, :2] -= shift
        rms_bbs[:, 2:] -= shift
        fp_eds[:, :2] -= shift
        fp_eds[:, 2:] -= shift
        tl -= shift
        br -= shift
        eds_to_rms_tmp = []
        for l in range(len(eds_to_rms)):
            eds_to_rms_tmp.append([eds_to_rms[l][0]])
        return rms_type, fp_eds, rms_bbs, eds_to_rms, eds_to_rms_tmp
