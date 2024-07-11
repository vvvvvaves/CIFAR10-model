import datetime
from training_loop import training_loop
from val_loop import validate
import pandas as pd
import torch


def train(epochs, train_loader, val_loader, model, optimizer, loss_fn):

    history = pd.DataFrame(columns=['datetime',
                                    'epoch',
                                    'train_accuracy',
                                    'train_loss_per_batch',
                                    'val_accuracy',
                                    'val_loss_per_batch'])

    for epoch in range(1, int(epochs) + 1):
        _datetime = datetime.datetime.now()
        print(f"{_datetime} Epoch {epoch}: ")
        training_loop(optimizer, model, loss_fn, train_loader)
        stats = validate(model, train_loader, val_loader, loss_fn)

        history.loc[epoch - 1] = [_datetime, epoch] + stats

        result = {'stats': [_datetime, epoch] + stats,
                  'model': model.state_dict()}

        torch.save(result, f"./result_e{epoch}.pt")

    return history

