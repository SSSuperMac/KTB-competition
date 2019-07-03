import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from net_utils import convolution, residual
from net_utils import make_layer, make_layer_revr
from net_utils import make_pool_layer, make_unpool_layer
from net_utils import make_hg_layer, make_merge_layer, _sigmoid


class kp_module(nn.Module):
    def __init__(
        self, n, dims, modules, layer=residual,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, **kwargs
    ):
        super(kp_module, self).__init__()

        self.n   = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1  = make_up_layer(
            3, curr_dim, curr_dim, curr_mod, 
            layer=layer, **kwargs
        )  
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.low2 = kp_module(
            n - 1, dims[1:], modules[1:], layer=layer, 
            make_up_layer=make_up_layer, 
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            **kwargs
        ) if self.n > 1 else \
        make_low_layer(
            3, next_dim, next_dim, next_mod,
            layer=layer, **kwargs
        )
        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.up2  = make_unpool_layer(curr_dim)

        self.merge = make_merge_layer(curr_dim)

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return self.merge(up1, up2)

class Backbone(nn.Module):
    def __init__(self, downsample_image=False, pre_dim=64, feat_dim=256):
        n = 4
        dims    = [pre_dim, 128, 256, 256, 384, 512]
        modules = [2, 2, 2, 2, 2, 4]

        stride_pre = 2 if downsample_image else 1
        self.pre = nn.Sequential(
            convolution(7, 1, 64, stride=stride_pre),
            residual(3, 64, pre_dim, stride=1)
            )

        self.kps = kp_module(
                n, dims, modules, layer=residual,
                make_up_layer=make_layer,
                make_low_layer=make_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            )

        self.conv = convolution(3, dims[n], feat_dim)

    def forward(self,x):
        y = self.pre(x)
        y = self.kps(y)
        out = self.conv(y)
        return out

class InfraredNet(nn.Module):
    def __init__(self, cnv_dim=256, curr_dim=256, out_dim=2):
        self.backbone = Backbone()

        self.heat_layer = nn.Sequential(
            convolution(3, cnv_dim, curr_dim, with_bn=False),
            nn.Conv2d(curr_dim, out_dim, (1, 1))
            )

    def forward(self, x):
        feature = self.backbone(x) # the same size with the input image
        heat_map = self.heat_layer(feature)
        return heat_map
