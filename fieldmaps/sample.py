"""Visualize sampling and resampling"""
from torchvision import datasets, transforms
import rbf
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

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


def main():
    query_points = torch.rand(28 * 28, 2) * 28
    grid_points = rbf.tensor_to_field_coords(28, 28)
    # query_points = query_points.numpy()
    # plot_query_points(query_points)

    train_loader = mnist_batch()
    for batch_idx, (data, target) in enumerate(train_loader):
        # import pdb; pdb.set_trace()
        img = data[0, 0]
        weights = rbf.mono_img_batch_to_weights(data)
        transformed = rbf.torch_rbf(query_points.unsqueeze(0), grid_points.unsqueeze(0), weights)

        # plt.imshow(img.numpy())
        plt.figure()
        plt.imshow(transformed.numpy().reshape(28, 28))
        plt.show()
    # plot the query points

main()
