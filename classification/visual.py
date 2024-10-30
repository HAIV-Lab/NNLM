from sklearn.manifold import TSNE
from model import MyModel
from build import build_dataset
from dataset import PrefixDataset1

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt


def Visual(args):
    my_model = MyModel(args=args)
    my_model = my_model.cuda()
    file_path = os.path.join(args.save_path, 'best_checkpoint.pth')
    best_checkpoint = torch.load(file_path)
    my_model.load_state_dict(best_checkpoint['best_model'])

    tsne = TSNE(n_components=2)
    dataset = PrefixDataset1(args=args)

    choice_label = np.arange(args.voc_len)
    np.random.shuffle(choice_label)

    real_choice_label = []
    for i in choice_label:
        if len(dataset.label[dataset.label==i]) >= 300:
            real_choice_label.append(i)
        if len(real_choice_label) == 2:
            break

    label = None
    embedding = None
    for i in real_choice_label:
        if label is None:
            label = dataset.label[dataset.label==i]
            embedding = dataset.data[dataset.label==i]
        else:
            label = np.concatenate((label, dataset.label[dataset.label==i]))
            embedding = np.concatenate((embedding, dataset.data[dataset.label==i]))

    embedding = torch.tensor(embedding).cuda()
    embedding = my_model.encode(embedding)
    embedding = embedding.cpu().detach().numpy()
    tsne_embedding = tsne.fit_transform(embedding)

    plt.scatter(tsne_embedding[:,0], tsne_embedding[:,1], c=label)
    plt.savefig("/data/zqh/adaptive-knn-mt/classification/images/koran.png")


