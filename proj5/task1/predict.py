import cv2
import torch
import matplotlib.pyplot as plt
import sys
import model
import numpy as np


def sample_test(network):
    """
    make the prediction for the first 10 images in test dataset. For each example image,
    print out the output values, index of the max output value, and the correct label of the digit. Furthermore,
    plot the first 9 digits as a 3x3 grid with the prediction for each example above it.
    :param network: the trained network
    """
    # get the first 10 digits and models from the test dataset, make the prediction on these digits.
    samples = next(iter(network.test_loader))
    images, labels = samples
    ground_truth_labels = labels[:10].numpy()
    output = network(images[:10])

    # for the first 9 digits, plot the digits and prediction. For all 10 digits, print the required results.
    for i in range(10):
        if i != 9:
            plt.subplot(3, 3, i + 1)
            plt.tight_layout()
            plt.imshow(images[i][0], cmap='gray', interpolation='none')
            plt.title("Prediction: {}".format(
                output.data.max(1, keepdim=True)[1][i].item()))
            plt.xticks([])
            plt.yticks([])
        output_values = ["{:.2f}".format(v) for v in output[i].tolist()]
        print(
            f"image {i + 1}, output values: {output_values}, max value index: {torch.argmax(output[i]).item()}, ground truth label: {ground_truth_labels[i]}")
    plt.show()


def new_input_test(network):
    """
    Read the digits from test_digits folder, preprocess them and feed into the network to do the prediction,
    plot the results.
    :param network: the neural network
    """
    file_path = './test_digits/'
    test_img = []
    for i in range(10):
        # load as grayscale image
        img = cv2.imread(f'{file_path}{i}.png', cv2.IMREAD_GRAYSCALE)
        # resize to 28 * 28 square
        img = cv2.resize(img, (28, 28))
        # invert the intensity since the new input images are black digits on white background
        img = cv2.bitwise_not(img)
        test_img.append(img)
    test_data = np.array(test_img)
    # expand the dimension and transform into tensor as input
    test_data = np.expand_dims(test_data, 1)
    test_data = torch.Tensor(test_data)
    output = network(test_data)

    for i in range(10):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(test_img[i], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def main(argv):
    """
    Read the network and run it on the test set, test the network on new inputs.
    :param argv: command line arguments
    """
    # load the model and enter eval mode
    network = model.MyNetwork()
    network.load_state_dict(torch.load('./results/model.pth'))
    network.eval()

    sample_test(network)
    new_input_test(network)


if __name__ == "__main__":
    main(sys.argv)
