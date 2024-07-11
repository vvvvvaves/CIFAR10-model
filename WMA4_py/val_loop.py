import torch
from torch.utils.data import DataLoader
from net import Net
from torch import nn


def validate(model: Net,
             train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
             val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
             loss_fn: nn.Module) -> list[float]:

    model.eval()
    stats = []
    for name, loader in [('[Train]', train_loader), ('[Val]', val_loader)]:
        correct = 0
        total = 0
        loss = 0.0
        with torch.no_grad():
            for imgs, labels in loader:
                outputs = model(imgs)
                preds = outputs.max(1)[1]

                total += labels.shape[0]
                correct += preds.eq(labels).sum().item()

                loss += loss_fn(outputs, labels).item()

        stats.append(round(correct / total * 100., 4))
        stats.append(round(loss / len(loader), 4))
        print(f"{name} Accuracy: {round(correct / total * 100., 4)}%,"
              + f" loss per batch: {round(loss / len(loader), 4)}")
    return stats