import torch
import webcolors
from PIL import Image, ImageDraw

from minimal.layout import NODE_COLOR

def draw_plan(
    masks, # torch.tensor of size (R, 64, 64) (torch.float32)
    nodes,  # list[int] of length R
    img_size=256
):
    # Convert masks to binary
    bin_masks = (masks > 0).byte()

    plan_img = Image.new("RGB", (64, 64), (255, 255, 255))
    draw = ImageDraw.Draw(plan_img)

    for m, n in zip(bin_masks, nodes):
        mask_bitmap = Image.fromarray(m.numpy() * 255, mode="L")
        r, g, b = webcolors.hex_to_rgb(NODE_COLOR[n])
        draw.bitmap((0, 0), mask_bitmap, fill=(r, g, b))

    return plan_img.resize((img_size, img_size), Image.Resampling.BOX)