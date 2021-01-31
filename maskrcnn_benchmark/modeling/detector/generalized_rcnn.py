# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..pwc.pwc import build_pwc_net
from ..attention.magnitude_attention import build_ma
from ..attention.direction_attention import build_da, create_worse_direction_matrix
from ..attention.channel_attention import build_ca

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.pwc_model = build_pwc_net()
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.previous_frame = None
        self.worse_direction_matrices = None
        self.MA = build_ma()
        self.DA = build_da()
        self.CA = build_ca()
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images = to_image_list(images)
        batch_size = len(images.tensors)
        prev_is_none = self.previous_frame is None

        # this is the backone where we get the features
        features = self.backbone(images.tensors)
        # Features shapes from pyramid backbone
        features_shape = [feat.shape for feat in features]
        '''
        # Creation of worse direction matrix for direction attention
        if self.worse_direction_matrices is None:
            self.worse_direction_matrices = [create_worse_direction_matrix(shape[2:]).cuda() for shape in features_shape]

        # Output handler for pwc result
        pyramid_layers_num = len(features_shape)
        flow_magn_dir = [torch.zeros([batch_size, 2, *shape[2:]]).cuda() for shape in features_shape]
        flow_features = []

        for i in range(batch_size):
            if i + 1 <= batch_size and i != 0:
                pwc_flow_features = self.pwc_model(images.tensors[i - 1], images.tensors[i], features_shape)
            else:
                if prev_is_none:
                    pwc_flow_features = self.pwc_model(images.tensors[i], images.tensors[i], features_shape)
                else:
                    pwc_flow_features = self.pwc_model(self.previous_frame, images.tensors[i], features_shape)

            for j in range(pyramid_layers_num):
                flow_magn_dir[j][i, :, :, :] = pwc_flow_features['fpn_layers'][j]['tenFlow']
                if j >= len(flow_features):
                    flow_features.append(pwc_flow_features['fpn_layers'][j]['tenFeat'])
                else:
                    flow_features[j] = torch.stack([flow_features[j],
                                                    pwc_flow_features['fpn_layers'][j]['tenFeat']]).squeeze()

            del pwc_flow_features

        if not self.CA.is_in_channels_set:
            flow_channels_num = [flow_layer.shape[1] for flow_layer in flow_features]
            self.CA.set_convolutional_layers(flow_channels_num)

        # Magnitude attention
        features_MA = self.MA(flow_magn_dir, features, batch_size, prev_is_none)
        # Direction attention
        features_DA = self.DA(flow_magn_dir, features, self.worse_direction_matrices, batch_size, prev_is_none)
        # Channel attention
        #features_CA = self.CA(flow_features, features, prev_is_none)

        # Store for new iteration
        self.previous_frame = images.tensors[-1]

        #features = [features_SA[i].detach() for i in range(pyramid_layers_num)]

        # Element-wise addition
        features = [features_MA[i].detach() + features_DA[i].detach() for i in range(pyramid_layers_num)]
        '''
        #  this is the RPN network where we pass the features.
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
