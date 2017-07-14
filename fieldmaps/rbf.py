"""Radial Basis Function"""
import torch
from torch.autograd import Function, Variable, gradcheck
import numpy as np

def gaussian(r, eta=0.5):
    "Gaussian radial basis function"
    p = (eta*r)
    pp = p * p
    return torch.exp(-pp)


def gauss_deriv(r, eta=1.0):
    "Derivative of gaussian wrt to r"
    # TODO: Expect to incldue eta
    return -2 * torch.exp(-(r*r)) * r

def euclid(x, y):
    d = x - y
    sqrd = d * d
    dsum = torch.sum(sqrd)
    dsum = torch.FloatTensor([dsum])
    return torch.sqrt(dsum)

def batch_euclid(x, y):
    batch_size = x.size(0)
    d = x - y
    sqrd = d * d
    dsum = torch.sum(sqrd, 2)
    return torch.sqrt(dsum)

def tensor_to_field_coords(h, w, shift=False):
    """Convert integer coordinates of an image into points"""
    coords = []
    for i in range(h):
        for j in range(w):
            if shift:
                coords.append([i+0.5, j+0.5])
            else:
                coords.append([i, j])

    return torch.FloatTensor(coords)

def tensor_to_field_weights(data):
    if data.dim() != 2:
        raise ValueError
    h, w = data.size()
    weights = []
    for i in range(h):
        for j in range(w):
            weights.append(data[i, j])

    return torch.FloatTensor(weights)

def mono_img_batch_to_weights(img_batch):
    if img_batch.dim() != 4:
        raise ValueError
    batch_size, nchannels, h, w = img_batch.size()
    field_batch = torch.zeros(batch_size, h*w)
    if img_batch.is_cuda:
        field_batch = field_batch.cuda()
    for i in range(batch_size):
        a = tensor_to_field_weights(img_batch[0,0,:,:])
        field_batch[i] = a

    return field_batch



def torch_rbf(input, points, weights):
    radialf = gaussian
    dist = batch_euclid
    # Check batch size consistent
    if not(points.size(0) == weights.size(0) == input.size(0)):
        raise ValueError

    if points.size(1) != weights.size(1):
        raise ValueError

    if input.size(2) != points.size(2):
        raise ValueError

    batch_size = points.size(0)
    npoints = weights.size(1)
    nquery_points = input.size(1)
    ndim = input.size(2)

    # final_dists = torch.zeros(batch_size, nquery_points)
    final_dists = Variable(torch.zeros(batch_size, nquery_points))
    if input.is_cuda:
        final_dists = final_dists.cuda()

    for i in range(npoints):
        point_slice = points[:, i:i+1, :].expand_as(input)
        # Distance from each input query point to ith rbf center
        dists = dist(input, point_slice).squeeze(2)
        if i == 1:
            print("DIST", dists[0,0])
        rad_dists = radialf(dists)
        tile_weights = weights[:,i:i+1].expand_as(rad_dists)
        w_rad_dists = tile_weights * rad_dists
        final_dists = final_dists + w_rad_dists

    return final_dists


# Inherit from Function
class RBF(Function):

    # bias is an optional argument
    def forward(self, input, points, weights):
        """
        Computes radial basis function:

        s(x) = sum([weight[i] * radialf(dist(input, points[i])) for i = 1:npoints])

        input: [ndim] - input points
        points:: [batch_size, npoints, ndim] - vector of points in n-dim space
        weights: [batch_size, npoints]
        dist: distance between two points, e.g. Euclidean distance
        radialf:: R -> R: radial function
        """
        radialf = gaussian
        dist = batch_euclid
        if points.size(0) != weights.size(0):
            raise ValueError

        if points.size(1) != weights.size(1):
            raise ValueError

        if input.size(0) != points.size(2):
            raise ValueError

        batch_size = points.size(0)
        npoints = weights.size(1)
        tile_input = input.repeat(batch_size, npoints, 1)

        # Dist
        dists = dist(tile_input, points)
        rad_dists = radialf(dists)
        w_rad_dists = weights * rad_dists

        self.save_for_backward(input, points, weights)
        return torch.sum(w_rad_dists, 1)

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. input to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional input.
        input, points, weights = self.saved_tensors
        radialf = gaussian
        dist = batch_euclid


        # Do forward computation again (TODO: Cache)
        batch_size = points.size(0)
        npoints = weights.size(1)
        tile_input = input.repeat(batch_size, npoints, 1)

        # Dist
        dists = dist(tile_input, points)
        rad_dists = radialf(dists)
        w_rad_dists = weights * rad_dists

        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for input that don't require it is
        # not an error.

        deriv_radialf = gauss_deriv

        # dloss_dweights
        dloss_df = grad_output
        dloss_dweights = w_rad_dists * dloss_df.expand_as(w_rad_dists)

        # dloss_dinp
        deriv_rad_dists = deriv_radialf(dists)
        deriv_w_rad_dists = weights * deriv_rad_dists
        import pdb; pdb.set_trace()
        return dloss_dinp, dloss_dpoints, dloss_dweights
        #
        #
        #
        # if self.needs_input_grad[0]:
        #     grad_input = grad_output.mm(weight)
        # if self.needs_input_grad[1]:
        #     grad_weight = grad_output.t().mm(input)
        # if bias is not None and self.needs_input_grad[2]:
        #     grad_bias = grad_output.sum(0).squeeze(0)
        #
        # return grad_input, grad_weight, grad_bias


def rbf(input, points, weights):
    """Radial basis function"""
    return RBF()(input, points, weights)

def test_rbf_grads():
    test = gradcheck(Linear(), input, eps=1e-6, atol=1e-4)

def test_rbf():
    batch_size = 5
    npoints = 3
    nquery_points = 4
    ndim = np.random.randint(1,4)
    points = torch.randn(batch_size, npoints, ndim)
    points = Variable(points, requires_grad=True)

    weights = torch.randn(batch_size, npoints)
    weights = Variable(weights, requires_grad=True)

    # Say we are trying to do scrambed mnist
    # For each input image we create an rbf
    # Then we want to evaluate the field at multiple points in 2d
    # In this example we expect the same points for each element of the batch
    # but tha may not be the case in general
    # So input should be [batch_size, nquery_points, ndim]
    inputs = torch.randn(batch_size, nquery_points, ndim)
    inputs = Variable(inputs)
    res = torch_rbf(inputs, points, weights)
    print("Output is", res)
    loss = torch.mean(res)
    loss.backward()

# test_rbf()
