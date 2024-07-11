import os.path

import torch
from torchvision import datasets, transforms
import pickle


def get_data(transformations):
    data_path = '../data/'
    train_set_t = datasets.CIFAR10(data_path,
                                   train=True,
                                   download=False,
                                   transform=transformations)
    test_set_t = datasets.CIFAR10(data_path,
                                  train=False,
                                  download=False,
                                  transform=transformations)

    return train_set_t, test_set_t


def get_norms(train_set_t):
    if os.path.exists("norms.pickle"):
        with open("norms.pickle", "rb") as handle:
            norms = pickle.load(handle)
    else:
        if train_set_t is None:
            raise Exception("Train set is none.")
        imgs = [image for image, label in train_set_t]
        imgs_t = torch.stack(imgs, dim=3)
        means = imgs_t.view(3, -1).mean(dim=1)
        stds = imgs_t.view(3, -1).std(dim=1)
        norms = (means, stds)
        with open("norms.pickle", "wb") as handle:
            pickle.dump(norms, handle, pickle.HIGHEST_PROTOCOL)

    return norms



