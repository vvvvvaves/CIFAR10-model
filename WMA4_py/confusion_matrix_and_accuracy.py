import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import pickle

from net import Net
from data import *


def load_model():
    state = torch.load('./models/result_e8.pt')
    model = Net()
    model.load_state_dict(state['model'])
    return model


def predict(model, val_loader):
    model.eval()

    true_y = []
    pred_y = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            outputs = model(imgs)
            preds = outputs.max(1)[1]

            pred_y.append(preds)
            true_y.append(labels)

        true_y = torch.cat(true_y, dim=0)
        pred_y = torch.cat(pred_y, dim=0)

    return true_y, pred_y


def run():
    if os.path.exists("true_pred_y.pickle"):
        with open("true_pred_y.pickle", "rb") as handle:
            true_y, pred_y = pickle.load(handle)
    else:
        means, stds = get_norms(None)
        transformations = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=means, std=stds)])
        train_set_t, test_set_t = get_data(transformations=transformations)

        model = load_model()

        val_loader = DataLoader(test_set_t, batch_size=4096, shuffle=False)

        true_y, pred_y = predict(model, val_loader)

        with open("true_pred_y.pickle", "wb") as handle:
            pickle.dump((true_y, pred_y), handle, pickle.HIGHEST_PROTOCOL)

    cm = confusion_matrix(true_y, pred_y)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    ax = seaborn.heatmap(cm, annot=True, fmt='g')
    ax.xaxis.set_ticklabels(class_names, fontsize=10)
    ax.yaxis.set_ticklabels(class_names, fontsize=10)
    plt.xticks(rotation=40)
    plt.yticks(rotation=0)

    log = pd.read_csv("./models/log.csv")
    train_acc = log.loc[log.epoch == 8, 'train_accuracy'][7]
    val_acc = log.loc[log.epoch == 8, 'val_accuracy'][7]
    plt.title(f"train accuracy: {train_acc}, val accuracy: {val_acc}")
    plt.savefig('confm_accuracy.png')


run()

