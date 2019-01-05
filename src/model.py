import torch
import torch.nn as nn

Actor = nn.Sequential(
    nn.Linear(6400, 256, bias = False),
    nn.ReLU(),
    nn.Linear(256, 256, bias = False),
    nn.ReLU(),
    nn.Linear(256, 1, bias = False),
    nn.Sigmoid()
)