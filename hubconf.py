import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = "cuda" if torch.cuda.is_available() else "cpu"

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SDG(model.parameters(),lr = 1e-3)
