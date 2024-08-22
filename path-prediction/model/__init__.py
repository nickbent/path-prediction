import torch.nn as nn


def linear_block( in_size, out_size, relu = True, layer_norm = False):
    linear = [nn.Linear(in_size, out_size)]

    if relu:
        linear.append(nn.ReLU())
    if layer_norm:
        linear.append(nn.LayerNorm(out_size))
    
    return linear

def linear_sequential(sizes, relu_last = True, layer_norm = False, layer_norm_last = False):

    linear = []
    relu = True
    for ix, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
        if ix == len(sizes) -2:
            relu = relu_last
            layer_norm = layer_norm_last
        linear+=linear_block(in_size, out_size, relu = relu, layer_norm=layer_norm)
    
    return nn.Sequential(*linear)

