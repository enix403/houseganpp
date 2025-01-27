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
import cv2
from PIL import Image, ImageDraw

cv2.setNumThreads(0)

ID_COLOR = {
    0: "#EE4D4D",
    1: "#C67C7B",
    2: "#FFD274",
    3: "#BEBEBE",
    4: "#BFE3E8",
    5: "#7BA779",
    6: "#E87A90",
    7: "#FF8C69",
    9: "#1F849B",
    14: "#727171",
    15: "#785A67",
    16: "#D3A2C7",
}

# ===============================

def draw_plan_2(
    masks, # torch.tensor of size (R, 64, 64) (torch.float32)
    real_nodes,  # list[int] of length R
    img_size=256
):
    masks = (masks > 0).float() * 255
    masks = torch.nn.functional.interpolate(
        masks.unsqueeze(1),
        size=(img_size, img_size),
        mode="area"
    ).squeeze(1).byte()

    plan_img = Image.new("RGB", (img_size, img_size), (255, 255, 255))  # Semitransparent background.
    draw = ImageDraw.Draw(plan_img)

    for m, nd in zip(masks, real_nodes):
        mask_bitmap = Image.fromarray(m.cpu().numpy(), mode="L")
        r, g, b = webcolors.hex_to_rgb(ID_COLOR[nd])
        draw.bitmap((0, 0), mask_bitmap, fill=(r, g, b))

    return plan_img


def draw_plan(
    masks, # torch.tensor of size (R, 64, 64) (torch.float32)
    real_nodes, # list[int] of length R
    im_size=256
):
    room_imgs = masks.clone().numpy()

    plan_img = Image.new(
        "RGBA", (im_size, im_size), (255, 255, 255, 255)
    )  # Semitransparent background.

    for m, nd in zip(room_imgs, real_nodes):

        # threshold at 0
        m[m > 0] = 255
        m[m < 0] = 0

        # resize
        m_lg = cv2.resize(m, (im_size, im_size), interpolation=cv2.INTER_AREA)

        # pick color
        color = ID_COLOR[nd]
        r, g, b = webcolors.hex_to_rgb(color)

        # set drawer
        draw = ImageDraw.Draw(plan_img)

        # draw region
        m_pil = Image.fromarray(m_lg)
        draw.bitmap((0, 0), m_pil.convert("L"), fill=(r, g, b, 256))

        # draw contour
        # m_cv = m_lg[:, :, np.newaxis].astype("uint8")
        # ret, thresh = cv2.threshold(m_cv, 127, 255, 0)
        # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours = [c for c in contours if len(contours) > 0]
        # cnt = np.zeros((256, 256, 3)).astype("uint8")
        # cv2.drawContours(cnt, contours, -1, (255, 255, 255, 255), 1)
        # cnt = Image.fromarray(cnt)
        # draw.bitmap((0, 0), cnt.convert("L"), fill=(0, 0, 0, 255))

    return plan_img.resize((im_size, im_size))

