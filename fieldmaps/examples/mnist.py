"""MNIST Using fieldmaps"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
plt.ion()

from fieldmaps.util.logger import Logger
import rbf

# Globals for vizualisation
plt.figure()
field_viz = plt.imshow(np.random.rand(28, 28))
plt.figure()
rbf_viz = plt.imshow(np.random.rand(28, 28))


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        # query_points = rbf.tensor_to_field_coords(28, 28, True)
        query_points = torch.rand(28 * 28, 2)* 28
        self.query_points = Parameter(query_points, requires_grad=True)

    def forward(self, x):
        field_viz.set_data(x[0,0,:,:].data.numpy())
        points = mnist_field_coords
        weights = Variable(rbf.mono_img_batch_to_weights(x.data))
        batch_size = x.size(0)
        print("QUERY POINTS SUM", torch.sum(self.query_points))
        rr = rbf.torch_rbf(self.query_points.unsqueeze(0).expand(batch_size, 28 * 28, 2),
                      points.unsqueeze(0).expand(batch_size, 28*28, 2), weights)
        x = rr.view(batch_size, 1, 28, 28)
        rbf_viz.set_data(x[0,0,:,:].data.numpy())
        plt.draw()
        plt.pause(0.001)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def product(xs):
    perm_size = 1
    for dim in xs:
        perm_size = perm_size * dim
    return perm_size

def permute_image(image, perm):
    """Return a permutation of the image"""
    batch_size = image.size()[0]
    perm = perm.repeat(batch_size, 1)
    img_flat = image.view([batch_size, -1])
    img_flat_perm = torch.gather(img_flat, 1, perm)
    return img_flat_perm.view(image.size())

def permute_image_test():
    batch_size = 2
    size = (batch_size, 3, 3)
    data = torch.arange(0, product(size)).view(size)
    perm = torch.randperm(product(size[1:])).repeat(batch_size, 1)
    return permute_image(data, perm, batch_size)


height = 28
width = 28
perm = torch.randperm(height * width)
mnist_field_coords = Variable(rbf.tensor_to_field_coords(height, width), requires_grad=False)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Permute the data
        # data = permute_image(data, perm)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        gradients = loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        # data = permute_image(data, perm)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()

if __name__ == "__main__":
    main()
