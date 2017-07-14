"""Use fieldmap to unscramble"""

from __future__ import print_function
import torch
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from sample import mnist_batch
import rbf

draw = False
if draw:
    plt.ion()

def main(epochs=1000, log_interval=10):
    # Setup
    batch_size = 16
    height = 28
    width = 28
    if draw:
        plt.figure()
        field_viz = plt.imshow(np.random.rand(height, width))
        plt.figure()
        rbf_viz = plt.imshow(np.random.rand(height, width))
    # plt.figure()
    # plt.scatter(query_points[:,0], query_points[:,1])

    perm = torch.randperm(height * width)
    mnist_field_coords = Variable(rbf.tensor_to_field_coords(height, width),
                                  requires_grad=False)

    query_points = torch.rand(height * width, 2) * 28
    grid_points = rbf.tensor_to_field_coords(height, width)
    grid_points = Variable(grid_points)
    query_points = Variable(query_points, requires_grad=True)
    optimizer = optim.SGD([query_points], lr=1, momentum=0.1)

    # Minimize
    train_loader = mnist_batch(batch_size)

    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            weights = Variable(rbf.mono_img_batch_to_weights(data))
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            tile_query_points = query_points.unsqueeze(0).expand(batch_size,
                                height * width, 2)
            tile_grid_points = grid_points.unsqueeze(0).expand(batch_size,
                                height * width, 2)
            transformed = rbf.torch_rbf(tile_query_points, tile_grid_points,
                                        weights)
            transformed_img = transformed.view(batch_size, height, width)
            img_distance = transformed_img - data
            loss = torch.mean(torch.sqrt(img_distance*img_distance))
            print("Loss is", loss.data[0])
            loss.backward()
            optimizer.step()

            if draw:
                field_viz.set_data(data[0,0,:,:].data.numpy())
                rbf_viz.set_data(transformed_img[0,:,:].data.numpy())
                plt.draw()
                plt.pause(0.0001)

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))


if __name__ == "__main__":
    main()
