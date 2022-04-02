import os
import sys
import torch
from task1.model import MyNetwork
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd


# The truncated model inherited from MyNetwork implementation, it terminates at the Dense layer with 50 outputs.
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
    """
    Create the dataset from the greek symbols, save the labels for each symbol and corresponding intensity values.
    :param directory: the directory of the greek symbols dataset
    """
    # save the intensity values into data list
    data = []
    # save the corresponding category(alpha = 0, beta = 1, gamma = 2) into the categories list
    categories = []
    symbol_dict = {}
    i = 0
    for file_name in os.listdir(directory):
        plt.subplot(9, 3, i + 1)
        i += 1

        # keep track of the corresponding category
        symbol = file_name.split('_')[0]
        if symbol not in symbol_dict:
            symbol_dict[symbol] = len(symbol_dict)
        categories.append(symbol_dict[symbol])

        # read the image, resize and convert to grayscale, then invert the intensity
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

    # save the intensity values into data.csv and categories into category.csv
    data_header = [str(i) for i in range(len(data[0]))]
    data = pd.DataFrame(data)
    data.to_csv('./data.csv', header=data_header, index=False)

    categories = pd.DataFrame(categories)
    categories.to_csv('./category.csv', header=['category'], index=False)


def embedding_projection(network):
    """
    Load the greek dataset from the csv files, feed the network with the tensor format data to get the embedding
    :param network: the trained network with output size as 50
    :return: the projected embeddings and categories
    """
    with torch.no_grad():
        data = np.array(pd.read_csv('./data.csv'))
        data = data.reshape((data.shape[0], 1, 28, 28))
        categories = np.array(pd.read_csv('./category.csv')).flatten()
        embedding = network(torch.Tensor(data))
        return embedding.numpy(), categories


def ssd(a, b):
    """
    Calculate the sum squared distance between two embedding arrays
    :param a: one embedding array
    :param b: the other embedding array
    :return: the sum squared distance
    """
    dif = a.ravel() - b.ravel()
    return np.dot(dif, dif)


def dist_cal(categories, embedding):
    """
    For each category of all the categories, pick the first greek digit to calculate the sum squared distance from
    all the greek symbols, sort these three distance lists to show the pattern.
    :param categories: the category list representing each greek symbol file's category(0/1/2)
    :param embedding: the embedding got from the truncated model
    """
    category_first_idx = []
    dist_label_list = []
    labels = {0: 'alpha', 1: 'beta', 2: 'gamma'}
    # find the first greek symbol for each category
    for i in range(3):
        category_first_idx.append(np.where(categories == i)[0][0])
    for i in range(3):
        dist_label_list.append([])
        # for each category, calculate the distance between the first one and all the greek symbols
        for j in range(embedding.shape[0]):
            dist = ssd(embedding[category_first_idx[i]], embedding[j])
            dist_label_list[i].append((dist, labels[categories[j]]))
        # sort by the distance value
        dist_label_list[i].sort(key=lambda v: v[0])
        print(dist_label_list[i])


def test_greek(network, embedding, categories):
    """
    Read these images from the test_greek folder as test dataset, preprocess them and feed into the network to get
    the embeddings, calculate the distances between each test image and the whole training data's embedding,
    make the one with smallest distance as the prediction.
    :param network: trained truncated network
    :param embedding: the training dataset embeddings
    :param categories: the training dataset categories
    """
    labels = {0: 'alpha', 1: 'beta', 2: 'gamma'}
    file_path = './test_greek/'
    test_img = []
    for file_name in os.listdir(file_path):
        # load as grayscale image
        img = cv2.imread(file_path + file_name, cv2.IMREAD_GRAYSCALE)
        # resize to 28 * 28 square
        img = cv2.resize(img, (28, 28))
        # invert the intensity since the new input images are black digits on white background
        img = cv2.bitwise_not(img)
        test_img.append(img)
    test_data = np.array(test_img)
    # expand the dimension and transform into tensor as input
    test_data = np.expand_dims(test_data, 1)
    test_data = torch.Tensor(test_data)

    print("Test dataset result:")
    with torch.no_grad():
        output = network(test_data)
        test_data = [i for i in output.numpy()]
        for i in range(6):
            plt.subplot(3, 2, i + 1)
            plt.tight_layout()
            plt.imshow(test_img[i], cmap='gray', interpolation='none')
            result = []
            # for each image in the test dataset, calculate the SSD and sort
            for idx in range(len(categories)):
                result.append((ssd(embedding[idx], test_data[i]), labels[categories[idx]]))
            result.sort(key=lambda v: v[0])
            print(result)
            plt.title("Prediction: {}".format(result[0][1]))
            plt.xticks([])
            plt.yticks([])
        plt.show()


def main(argv):
    """
    Build the greek symbols dataset in the csv format, create the truncated model to calculate the embedding for the
    dataset, calculate the sum squared distance between each single example of each category and all the dataset. In
    the end, apply this embedding and SSD to a new test dataset to see the effect.
    :param argv: the command line parameters
    """
    create_dataset('./greek')

    truncated_model = TruncatedModel()
    truncated_model.load_state_dict(torch.load('../task1/results/model.pth'))
    truncated_model.eval()

    samples = next(iter(truncated_model.test_loader))
    images, labels = samples
    output = truncated_model(images[0:1])
    # show the size of output of the truncated model is 50
    assert output.shape[1] == 50

    embedding, categories = embedding_projection(truncated_model)

    dist_cal(categories, embedding)

    test_greek(truncated_model, embedding, categories)


if __name__ == "__main__":
    main(sys.argv)
