import datetime
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd

from data import *
from train import train
from net import Net


def run():
    means, stds = get_norms(None)
    transformations = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=means, std=stds)])
    train_set_t, test_set_t = get_data(transformations=transformations)

    epochs = 10
    train_loader = DataLoader(train_set_t, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(test_set_t, batch_size=64, shuffle=True, num_workers=4)
    model = Net()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True, weight_decay=1e-3)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.NLLLoss()
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    log = train(epochs, train_loader, val_loader, model, optimizer, loss_fn)
    # scheduler.step()

    log.to_csv("./log.csv")
    torch.save(model.state_dict(), "./model.pt")


if __name__ == '__main__':
    run()


