import sys
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot


# The network for the MNIST dataset classification. It contains:
# A convolution layer with 10 5x5 filters
# A max pooling layer with a 2x2 window and a ReLU function applied.
# A convolution layer with 20 5x5 filters
# A dropout layer with a 0.5 dropout rate (50%)
# A max pooling layer with a 2x2 window and a ReLU function applied
# A flattening operation followed by a fully connected Linear layer with 50 nodes and a ReLU function on the output
# A final fully connected Linear layer with 10 nodes and the log_softmax function applied to the output.
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        # initialize the configurations for this network
        self.n_epochs = 5
        self.batch_size_train = 64
        self.batch_size_test = 1000
        self.learning_rate = 0.01
        self.momentum = 0.5
        self.log_interval = 10

        self.random_seed = 42
        torch.backends.cudnn.enabled = False
        torch.manual_seed(self.random_seed)
        # load the training dataset and test dataset
        self.train_dataset = torchvision.datasets.MNIST(root='../data', train=True, download=True,
                                                        transform=torchvision.transforms.Compose([
                                                            torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize(
                                                                (0.1307,), (0.3081,))
                                                        ]))
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size_train)
        self.test_dataset = torchvision.datasets.MNIST('../data', train=False, download=True,
                                                       transform=torchvision.transforms.Compose([
                                                           torchvision.transforms.ToTensor(),
                                                           torchvision.transforms.Normalize(
                                                               (0.1307,), (0.3081,))
                                                       ]))
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size_test)

        # define the layers
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(5, 5))
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        computes a forward pass for the network
        :param x: feed data
        :return: 10 values indicating the confidence of the data towards labels
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def show_examples(network):
    """
    Plot the first 6 digits in the test dataset along with the ground truth labels.
    :param network: the neural network
    """
    examples = enumerate(network.test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def train_network(network, epoch, train_losses, train_counter):
    """
    Train the network on the training dataset, calculate the loss values.
    :param network: the network that is during training
    :param epoch: the training epoch
    :param train_losses: the training loss values in a list
    :param train_counter: the counter for training
    """
    optimizer = optim.SGD(network.parameters(), lr=network.learning_rate,
                          momentum=network.momentum)

    network.train()
    for batch_idx, (data, target) in enumerate(network.train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % network.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(network.train_loader.dataset),
                       100. * batch_idx / len(network.train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(network.train_loader.dataset)))
            torch.save(network.state_dict(), 'results/model.pth')
            torch.save(optimizer.state_dict(), 'results/optimizer.pth')


def test_network(network, test_losses):
    """
    Test the network performance on the test dataset, calculate the corresponding loss.
    :param network: the network that is during training
    :param test_losses: the loss values on the test dataset
    """
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in network.test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(network.test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(network.test_loader.dataset),
        100. * correct / len(network.test_loader.dataset)))


def main(argv):
    """
    Create the network, plot several images from the test dataset, save the diagram of this network as "diagram.png"
    file, train the network and plot the training and validation loss. :param argv: the command line parameters
    """
    network = MyNetwork()
    show_examples(network)

    # plot the diagram
    x = torch.randn(1000, 1, 28, 28)
    dot = make_dot(network(x), params=dict(network.named_parameters()))
    # make sure the Graphviz is defined in your system path, otherwise it will throw an exception. If you do not want to
    # install, comment this line
    dot.render('diagram', format='png')

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(network.train_loader.dataset) for i in range(network.n_epochs + 1)]

    test_network(network, test_losses)
    for epoch in range(1, network.n_epochs + 1):
        train_network(network, epoch, train_losses, train_counter)
        test_network(network, test_losses)

    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
