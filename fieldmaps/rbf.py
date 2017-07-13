"""Radial Basis Function"""
import torch
from torch.autograd import Function, Variable

def gaussian(r, eta=1.0):
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


def test_rbf():
    # I expect a batch of inputs, e.g. 64
    # each batch has a set of query points (or one now for simplicity)
    # But the rbf itself has one set of points, e.g. 12

    # So suppose we take in a set of images
    # Each image, i.e. element of the batch is converted into a field by some
    # deterministic fitting process
    # So for each element of the batch, I have a set of points and a set of weights

    # and now I want to query each onae of those fields to produce a new field
    # Suppose I query each one once, I get back a vector of batch_size long
    # Then each element of this vector becomes teh points value for another rbf

    # So will we ever be in the regime where we have one rbf
    # For example for a permutation, we are making a new rbf for every input,
    # And actually its the input values that are being learned

    # So it's definitely true that the number of point-sets might be greater than 1
    # And that number will typically be the batch_size

    # For each rbf we're going to index it n times,
    # Those inex positions might be the same, as in for a permutation network,
    # Or potentially they could be unique for each element of the batch
    # so n or 1 are acceptable sizes, for now we'll focus on 1
    batch_size = 5
    npoints = 3
    ndim = 1
    points = torch.randn(batch_size, npoints, ndim)
    points = Variable(points, requires_grad=True)

    weights = torch.randn(batch_size, npoints)
    weights = Variable(weights, requires_grad=True)

    input = torch.randn(ndim)
    input = Variable(input)
    res = rbf(input, points, weights)
    import pdb; pdb.set_trace()
    print("Output is", res)
    loss = torch.mean(res)
    loss.backward()

    # Training
    learning_rate = 1-e6
    n_iters = 500
    for t in range(n_iters):
        print("iter")

test_rbf()
