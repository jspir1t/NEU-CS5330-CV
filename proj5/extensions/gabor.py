import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from task1.model import MyNetwork
import torch.nn.functional as F
import torch.nn as nn
import sys


# Network with first convolutional layer replaced by a filter bank of Gabor.
class GaborConvNetwork(MyNetwork):

    def __init__(self, gabor_kernels, check_first_layer=False):
        super().__init__()
        self.gabor_kernels = gabor_kernels
        self.check_first_layer = check_first_layer

    # override the forward method
    def forward(self, x):
        gabor_conv = nn.Conv2d(1, 10, kernel_size=(5, 5))
        gabor_conv.weight = nn.Parameter(torch.Tensor(self.gabor_kernels))

        # if checking the result of first convolutional layer, set the parameter check_first_layer as True
        if self.check_first_layer:
            return gabor_conv(x)
        x = F.relu(F.max_pool2d(gabor_conv(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def generate_gabor_kernels():
    """
    Generate ten gabor kernels with different angles from 0 to 180 degree
    :return: the gabor kernels in a numpy array with the dimension as required
    """
    gabor_kernels = [cv2.getGaborKernel((5, 5), 10, np.degrees(i), 0.05, 0.05, 0, cv2.CV_32F) for i in
                     np.linspace(0, 180, 10)]
    gabor_kernels = np.array(gabor_kernels)
    gabor_kernels = np.expand_dims(gabor_kernels, 1)
    return gabor_kernels


def test_dataset_acc(network):
    """
    Feed the network with the test dataset to calculate the accuracy.
    :param network: the network with first convolutional layer replaced by a set of Gabor filters
    """
    correct = 0
    with torch.no_grad():
        for data, target in network.test_loader:
            output = network(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(network.test_loader.dataset),
                                                           100. * correct / len(network.test_loader.dataset)))


def first_layer_effect(weights, network, images):
    """
    Visualize the filters of first gabor convolutional layer and the corresponding output of the filter on the first
    test dataset image example
    :param weights: the values of the Gabor kernels
    :param network: the network with first convolutional layer replaced by a set of Gabor filters
    :param images: the test dataset images
    """
    network.eval()
    with torch.no_grad():
        output = network(images[0:1])
        img_list = []
        for i in range(10):
            img_list.append(weights[i, 0])
            img_list.append(output[0, i])

    fig, axs = plt.subplots(5, 4, figsize=(9, 6), subplot_kw={'xticks': [], 'yticks': []})
    for i in range(20):
        row = i // 4
        col = i % 4
        axs[row, col].imshow(img_list[i], cmap='gray')
    plt.show()


def main(argv):
    """
    Create a network inherited from MyNetwork and replace original first convolutional layer with a layer containing
    Gabor kernels. Check the prediction and first layer's output of the first test dataset example, calculate the
    accuracy of this network on the test dataset.
    :param argv: command line parameters
    """
    gabor_kernels = generate_gabor_kernels()

    network = GaborConvNetwork(gabor_kernels)
    network.load_state_dict(torch.load('../task1/results/model.pth'))
    network.eval()

    samples = next(iter(network.test_loader))
    images, labels = samples

    # make the prediction on the first image in the test dataset
    output = network(images[0:1])
    prediction = output.data.max(1, keepdim=True)[1].item()
    print(prediction)

    test_dataset_acc(network)

    first_layer_network = GaborConvNetwork(gabor_kernels, True)
    first_layer_network.load_state_dict(torch.load('../task1/results/model.pth'))
    first_layer_effect(gabor_kernels, first_layer_network, images)


if __name__ == "__main__":
    main(sys.argv)
