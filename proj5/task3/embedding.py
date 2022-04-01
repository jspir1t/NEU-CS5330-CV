import os
import sys
import torch
from task1.model import MyNetwork
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import csv
import numpy as np
import pandas as pd


class TruncatedModel(MyNetwork):

    def __init__(self):
        super().__init__()

    # override the forward method
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return x


def create_dataset(directory):
    data = []
    categories = []
    symbol_dict = {}
    i = 0
    for file_name in os.listdir(directory):
        plt.subplot(3, 9, i + 1)
        i += 1

        symbol = file_name.split('_')[0]
        if symbol not in symbol_dict:
            symbol_dict[symbol] = len(symbol_dict)
        categories.append(symbol_dict[symbol])

        img = cv2.imread(os.path.join(directory, file_name))
        img = cv2.resize(img, (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.bitwise_not(img)

        plt.tight_layout()
        plt.imshow(img, cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])

        data.append(np.array(img).flatten())
    plt.show()

    data_header = [str(i) for i in range(len(data[0]))]
    data = pd.DataFrame(data)
    data.to_csv('./data.csv', header=data_header, index=False)

    categories = pd.DataFrame(categories)
    categories.to_csv('./category.csv', header=['category'], index=False)


def embedding_projection(network):
    with torch.no_grad():
        data = np.array(pd.read_csv('./data.csv'))
        data = data.reshape((data.shape[0], 1, 28, 28))
        categories = np.array(pd.read_csv('./category.csv')).flatten()
        embedding = network(torch.Tensor(data))
        return embedding.numpy(), categories


def ssd(a, b):
    dif = a.ravel() - b.ravel()
    return np.dot(dif, dif)


def dist_cal(categories, embedding):
    category_first_idx = []
    dists = []
    for i in range(3):
        category_first_idx.append(np.where(categories == i)[0][0])
    for i in range(3):
        dists.append([])
        for j in range(embedding.shape[0]):
            dists[i].append(ssd(embedding[category_first_idx[i]], embedding[j]))
    plt.plot(dists[0], 'r', dists[1], 'b', dists[2], 'g')
    plt.show()
    return dists



# main function (yes, it needs a comment too)
def main(argv):
    create_dataset('./greek')

    network = MyNetwork()
    network.load_state_dict(torch.load('../task1/results/model.pth'))

    samples = next(iter(network.test_loader))
    images, labels = samples

    truncated_model = TruncatedModel()
    truncated_model.eval()
    output = truncated_model(images[0:1])
    assert output.shape[1] == 50

    embedding, categories = embedding_projection(truncated_model)

    dist_cal(categories, embedding)


if __name__ == "__main__":
    main(sys.argv)
