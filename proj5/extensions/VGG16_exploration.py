import cv2
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt

import sys


# Inherited from pretrained VGG-16 model with layers extracted in self.features, through which we could return at
# the specified layer.
class VGG16(torch.nn.Module):
    def __init__(self, layer_idx):
        super(VGG16, self).__init__()
        features = list(models.vgg16(pretrained=True).features)[:23]
        self.features = nn.ModuleList(features).eval()
        self.layer_idx = layer_idx
        # print(features)

    # override the forward method
    def forward(self, x):
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == self.layer_idx:
                return x


def first_layer_effect(network, img):
    """
    Plot the frst 64 filters' output of the VGG-16 network
    :param network: the pretrained network
    :param img: the image example
    """
    network.eval()
    with torch.no_grad():
        output = network(img)
        img_list = []
        for i in range(64):
            img_list.append(output[0, i])

    fig, axs = plt.subplots(8, 8, figsize=(28, 28), subplot_kw={'xticks': [], 'yticks': []})
    for i in range(64):
        row = i // 8
        col = i % 8
        axs[row, col].imshow(img_list[i], cmap='gray')
    plt.show()


def main(argv):
    """
    create the VGG-16 model, feed the 3-channel image to the model to get the effect of the first convolutional layer
    :param argv: command line parameters
    """
    model = VGG16(0)

    img = cv2.imread('./car.jpg')
    img = cv2.resize(img, (224, 224))

    img = np.array(img).astype('float32')
    tensor = torch.from_numpy(img)
    reshaped = tensor.permute(2, 0, 1).unsqueeze(0)

    first_layer_effect(model, reshaped)


if __name__ == "__main__":
    main(sys.argv)
