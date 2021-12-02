import math
from typing import List

import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np
from megengine import Tensor
from megengine.module.normalization import GroupNorm

from .max_filter import MaxFiltering
from .utils import ShapeSpec


class DeFCNHead(M.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        self.stride_list = cfg.stride
        self.norm_reg_targets = cfg.norm_reg_targets
        in_channels = input_shape[0].channels
        num_classes = cfg.num_classes
        num_convs = 4
        prior_prob = cfg.cls_prior_prob
        num_anchors = [cfg.num_anchors] * len(input_shape)

        assert (
            len(set(num_anchors)) == 1
        ), "not support different number of anchors between levels"
        num_anchors = num_anchors[0]

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                M.Conv2d(in_channels,
                         in_channels,
                         kernel_size=3,
                         stride=1,
                         padding=1))
            cls_subnet.append(GroupNorm(32, in_channels))
            cls_subnet.append(M.ReLU())
            bbox_subnet.append(
                M.Conv2d(in_channels,
                         in_channels,
                         kernel_size=3,
                         stride=1,
                         padding=1))
            bbox_subnet.append(GroupNorm(32, in_channels))
            bbox_subnet.append(M.ReLU())

        self.cls_subnet = M.Sequential(*cls_subnet)
        self.bbox_subnet = M.Sequential(*bbox_subnet)
        self.cls_score = M.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = M.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1
        )
        self.max3d = MaxFiltering(
            in_channels, kernel_size=cfg.filter_ks, tau=cfg.filter_tau, use_gn=cfg.use_gn
        ) if cfg.use_3dmf else M.identity()
        self.filter = M.Conv2d(
            in_channels, num_anchors * 1, kernel_size=3, stride=1, padding=1
        )

        # Initialization
        for modules in [
            self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred,
            self.max3d, self.filter
        ]:
            for layer in modules.modules():
                if isinstance(layer, M.Conv2d):
                    M.init.normal_(layer.weight, mean=0, std=0.01)
                    M.init.fill_(layer.bias, 0)
                if isinstance(layer, GroupNorm):
                    M.init.fill_(layer.weight, 1)
                    M.init.fill_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        M.init.fill_(self.cls_score.bias, bias_value)

        self.scale_list = mge.Parameter(
            np.ones(len(self.stride_list), dtype=np.float32))

    def forward(self, features: List[Tensor]):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, K, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the K object classes.
            offsets (list[Tensor]): #lvl tensors, each has shape (N, 4, Hi, Wi).
                The tensor predicts 4-vector (dl,dt,dr,db) box
                regression values for every shift. These values are the
                relative offset between the shift and the ground truth box.
            filter (list[Tensor]): #lvl tensors, each has shape (N, 1, Hi, Wi).
        """
        logits, offsets, filter_subnet = [], [], []
        for feature, scale, stride in zip(features, self.scale_list, self.stride_list):
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_subnet = self.bbox_subnet(feature)
            offset_pred = self.bbox_pred(bbox_subnet) * scale
            if self.norm_reg_targets:
                offsets.append(F.relu(offset_pred) * stride)
            else:
                offsets.append(F.exp(offset_pred))
            filter_subnet.append(bbox_subnet)
        filters = [self.filter(x) for x in self.max3d(filter_subnet)]

        return logits, offsets, filters
