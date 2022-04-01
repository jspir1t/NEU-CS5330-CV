import sys
import torch
from task1.model import MyNetwork
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2

# The truncated model inherited from MyNetwork implementation, it only contains first one or first two layers of the
# original one.
class SubModel(MyNetwork):

    def __init__(self, layer_num):
        super().__init__()
        self.layer_num = layer_num

    # override the forward method, the layer_num specify one or two layers should be from the original network.
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # relu on max pooled results of conv1
        if self.layer_num == 2:
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # relu on max pooled results of dropout of conv2
        return x


def first_layer_analyze(network):
    """
    Visualize the ten filters using pyplot
    :param network: the trained network
    """
    network.eval()
    with torch.no_grad():
        weights = network.conv1.weight
        fig, axs = plt.subplots(3, 4, figsize=(9, 6), subplot_kw={'xticks': [], 'yticks': []})
        for i in range(10):
            row = i // 4
            col = i % 4
            axs[row, col].imshow(weights[i, 0].numpy())
            axs[row, col].set_title(f'Filter {i}')

    axs[2, 2].set_visible(False)
    axs[2, 3].set_visible(False)
    plt.show()


def filter_effect(network, img):
    """
    Apply OpenCV filter2D function to apply the 10 filters to the first training example image. Plot the result of
    each filter and the corresponding filter together.
    :param network: the trained network
    :param img: the first training example image
    """
    network.eval()
    with torch.no_grad():
        img_list = []
        weights = network.conv1.weight
        for i in range(10):
            img_list.append(weights[i, 0].numpy())
            filtered_img = cv2.filter2D(img.numpy()[0], -1, weights[i, 0].numpy())
            img_list.append(filtered_img)

    fig, axs = plt.subplots(5, 4, figsize=(9, 6), subplot_kw={'xticks': [], 'yticks': []})
    for i in range(20):
        row = i // 4
        col = i % 4
        axs[row, col].imshow(img_list[i], cmap='gray')
    plt.show()


def build_truncated_model(network, images):
    """
    Feed the first example image to the truncated model, plot the first 10 filters' output.
    :param network: the truncated model
    :param images: the first example image
    """
    network.eval()
    with torch.no_grad():
        output = network(images[0:1])
        fig, axs = plt.subplots(5, 2, figsize=(9, 6), subplot_kw={'xticks': [], 'yticks': []})
        for i in range(10):
            row = i // 2
            col = i % 2
            axs[row, col].imshow(output[0, i].numpy(), cmap='gray')
        plt.show()


def main(argv):
    """
    Load the network, plot the filters' weights of the first layer, shows the effects those filters on the example
    image. Then create two truncated models, one with first layer and the other with the first two layers,
    plot their result on the example image.
    :param argv: the command line input parameters
    """
    network = MyNetwork()
    network.load_state_dict(torch.load('../task1/results/model.pth'))

    samples = next(iter(network.train_loader))
    images, labels = samples
    img = images[0]

    first_layer_analyze(network)

    filter_effect(network, img)

    one_layer_model = SubModel(1)
    one_layer_model.load_state_dict(torch.load('../task1/results/model.pth'))
    build_truncated_model(one_layer_model, images)

    two_layer_model = SubModel(2)
    two_layer_model.load_state_dict(torch.load('../task1/results/model.pth'))
    build_truncated_model(two_layer_model, images)


if __name__ == "__main__":
    main(sys.argv)
