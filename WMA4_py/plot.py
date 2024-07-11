import datetime

import torch
import matplotlib.pyplot as plt
import pandas as pd


def plot():
    log = pd.read_csv('./models/log.csv')
    previous = None
    seconds = []
    for time in pd.to_datetime(log['datetime']):
        if previous is None:
            previous = time
            continue

        duration = time - previous
        seconds.append(duration.total_seconds())

        previous = time

    avg_duration = torch.mean(torch.tensor(seconds))

    plt.plot(log['epoch'], log['train_accuracy'])
    plt.plot(log['epoch'], log['val_accuracy'])
    plt.title("Training history")
    plt.xlabel(f"Average minutes/epoch: {round(avg_duration.item()/60,2)}")
    plt.savefig("training_history_plot.png")
    plt.show()


plot()