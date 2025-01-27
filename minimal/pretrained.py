import torch
from minimal.arch import Generator

_PRETRAINED_PATH = "./checkpoints/pretrained.pth"

model = Generator()

model.load_state_dict(
    torch.load(_PRETRAINED_PATH, map_location=torch.device("cpu")), strict=True
)

model = model.eval()
