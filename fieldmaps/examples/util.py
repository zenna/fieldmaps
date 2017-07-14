import torch

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
