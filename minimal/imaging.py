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

import numpy as np
import torch
import webcolors
import networkx as nx
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

cv2.setNumThreads(0)

ROOM_CLASS = {
    "living_room": 1,
    "kitchen": 2,
    "bedroom": 3,
    "bathroom": 4,
    "balcony": 5,
    "entrance": 6,
    "dining room": 7,
    "study room": 8,
    "storage": 10,
    "front door": 15,
    "unknown": 16,
    "interior_door": 17,
}

ID_COLOR = {
    1: "#EE4D4D",
    2: "#C67C7B",
    3: "#FFD274",
    4: "#BEBEBE",
    5: "#BFE3E8",
    6: "#7BA779",
    7: "#E87A90",
    8: "#FF8C69",
    10: "#1F849B",
    15: "#727171",
    16: "#785A67",
    17: "#D3A2C7",
}

# ===============================

def draw_plan(masks, real_nodes, im_size=256):
    room_imgs = masks.clone().numpy()

    bg_img = Image.new(
        "RGBA", (im_size, im_size), (255, 255, 255, 255)
    )  # Semitransparent background.

    for m, nd in zip(room_imgs, real_nodes):

        # resize map
        m[m > 0] = 255
        m[m < 0] = 0
        m_lg = cv2.resize(m, (im_size, im_size), interpolation=cv2.INTER_AREA)

        # pick color
        color = ID_COLOR[nd + 1]
        r, g, b = webcolors.hex_to_rgb(color)

        # set drawer
        dr_bkg = ImageDraw.Draw(bg_img)

        # draw region
        m_pil = Image.fromarray(m_lg)
        dr_bkg.bitmap((0, 0), m_pil.convert("L"), fill=(r, g, b, 256))

        # draw contour
        # m_cv = m_lg[:, :, np.newaxis].astype("uint8")
        # ret, thresh = cv2.threshold(m_cv, 127, 255, 0)
        # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours = [c for c in contours if len(contours) > 0]
        # cnt = np.zeros((256, 256, 3)).astype("uint8")
        # cv2.drawContours(cnt, contours, -1, (255, 255, 255, 255), 1)
        # cnt = Image.fromarray(cnt)
        # dr_bkg.bitmap((0, 0), cnt.convert("L"), fill=(0, 0, 0, 255))

    return bg_img.resize((im_size, im_size))

