import torch


class CA(torch.nn.Module):

    def __init__(self):
        super(CA, self).__init__()
        self.is_in_channels_set = False
        self.convFlow = []
        self.convOut = []

    def set_convolutional_layers(self, flow_channels_num):
        for num in flow_channels_num:
            # self.convFlow.append(torch.nn.Conv2d(in_channels=num, out_channels=256, kernel_size=3, stride=1,
            #                                     padding=2, dilation=2).cuda())
            self.convOut.append(torch.nn.Conv2d(in_channels=num + 256, out_channels=256, kernel_size=3, stride=1,
                                                padding=2, dilation=2).cuda())
        self.is_in_channels_set = True

    def forward(self, flow_features, image_features, prev_is_none):
        output_features = []
        for flowFeature, imageFeature, convOutLayer in zip(flow_features,
                                                           image_features,
                                                           # self.convFlow,
                                                           self.convOut):
            conv_out = convOutLayer(torch.cat([flowFeature, imageFeature], 1))
            if prev_is_none:
                conv_out[0, :, :, :] = imageFeature[0, :, :, :]
            output_features.append(conv_out)
        return output_features


def build_ca():
    ca = CA()
    return ca
