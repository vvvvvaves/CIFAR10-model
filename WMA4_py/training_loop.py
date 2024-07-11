import torch
from torch.utils.data import DataLoader
from net import Net


def training_loop(optimizer: torch.optim.Optimizer,
                  model: Net,
                  loss_fn: torch.nn.Module,
                  train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]) -> None:
    model.train()
    loss_train = 0.0
    correct = 0
    setlen = 0
    for imgs, labels in train_loader:
        outputs = model(imgs)

        loss = loss_fn(outputs, labels)

        preds = outputs.max(1)[1]
        correct += preds.eq(labels).sum().item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        loss_train += loss.item()

        setlen += len(train_loader)