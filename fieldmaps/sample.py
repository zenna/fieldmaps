"""Visualize sampling and resampling"""
from torchvision import datasets, transforms
import rbf
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

plt.ion()


def mnist_batch(batch_size=1):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    return train_loader

def plot_query_points(query_points):
    plt.scatter(query_points[:,0], query_points[:,1])


def main(niters=100000):
    query_points = torch.rand(28 * 28, 2) * 28
    grid_points = rbf.tensor_to_field_coords(28, 28)
    grid_points = Variable(grid_points)
    query_points = Variable(query_points, requires_grad=True)
    optimizer = optim.SGD([query_points], lr=1, momentum=0.1)

    # Minimize
    dist = query_points - grid_points
    loss = torch.mean(dist * dist)
    # query_points = query_points.numpy()
    # plot_query_points(query_points)

    train_loader = mnist_batch()
    for batch_idx, (data, target) in enumerate(train_loader):
        # import pdb; pdb.set_trace()
        img = data[0, 0]
        weights = rbf.mono_img_batch_to_weights(data)

        myobj = plt.imshow(np.random.rand(28, 28))
        for i in range(niters):
            transformed = rbf.torch_rbf(query_points.data.unsqueeze(0), grid_points.data.unsqueeze(0), weights)
            # plt.imshow(img.numpy())
            # plt.figure()
            img_data = transformed.numpy().reshape(28, 28)
            # img_data = np.random.rand(28, 28)
            if i % 100 == 0:
                myobj.set_data(img_data)
                plt.draw()
                plt.pause(0.001)
            # plt.imshow(transformed.numpy().reshape(28, 28))
            # plt.show()
            optimizer.zero_grad()
            dist = query_points - grid_points
            loss = torch.mean(dist * dist)
            print("Lossy", loss)
            loss.backward()
            optimizer.step()

    # plot the query points

main()
