import torch

"""
MIT License
Copyright (c) 2022 Abdullah Hamdi
Architecture adapted from MVTorch (https://github.com/ajhamdi/mvtorch).
"""

def batch_tensor(tensor, dim=1, squeeze=False):
    batch_size, dim_size = tensor.shape[0], tensor.shape[dim]
    returned_size = list(tensor.shape)
    returned_size[0] = batch_size * dim_size
    returned_size[dim] = 1
    if squeeze:
        return tensor.transpose(0, dim).reshape(returned_size).squeeze_(dim)
    else:
        return tensor.transpose(0, dim).reshape(returned_size)


def unbatch_tensor(tensor, batch_size, dim=1, unsqueeze=False):
    fake_batch_size = tensor.shape[0]
    nb_chunks = int(fake_batch_size / batch_size)
    if unsqueeze:
        return torch.cat(torch.chunk(tensor.unsqueeze_(dim), nb_chunks, dim=0), dim=dim).contiguous()
    else:
        return torch.cat(torch.chunk(tensor, nb_chunks, dim=0), dim=dim).contiguous()
